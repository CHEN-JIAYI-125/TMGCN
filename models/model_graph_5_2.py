#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/6/8

"""PyG-operator heterogeneous stance tracer (v5).

Topologies (built by datasets.dataset_graph_4, PyG-standard):

  - speaker hypergraph    : speaker_hyperedge_index [2, E_s] COO incidence
  - dialogue hypergraph   : dialogue_hyperedge_index [2, E_d] COO incidence
  - binary relation graph : temporal_edge_index + reply_candidate_edge_index
                            (relation 0 = temporal, 1 = reply candidate)

Convs are PyG standard operators (no hand-rolled scatter loops):

  - `HypergraphConv` on each hyperedge family (own parameters per family);
    `use_attention=True` gives attention within each hyperedge.
  - `RGCNConv` on the binary relation graph, with per-relation receive-side
    degree normalization passed as `edge_weight` so dense reply-candidate
    neighborhoods do not explode magnitudes.

Per layer i (h = layer output, messages are pure, residual added once):

    m_spk   = SpeakerHypergraphConv_i(h)
    m_dlg   = DialogueHypergraphConv_i(h)
    m_temp  = RGCNConv_i(h; temporal edges only, num_relations=1)
    m_reply = RGCNConv_i(h; reply-candidate edges only, num_relations=1)
    g       = softmax(Gate_i([h, sum of active m]))               # [N, 4]
    fused   = 4 * (g_spk*m_spk + g_dlg*m_dlg + g_temp*m_temp + g_reply*m_reply)
    h       = LayerNorm_i(h + fused)

Unlike v5 -- where temporal+reply shared a single RGCNConv and only their
SUM was visible to the gate -- each binary relation is now its own fusion
channel with independent parameters, exactly like the two hyperedge
families. The gate weight vector g = [g_spk, g_dlg, g_temp, g_reply] is
emitted per node ("fusion_gate" in the output dict), which is what the
case study reads to see how much attention each graph receives.

The gate head is zero-initialized, so at init the gates are 1/3 and the
block reduces exactly to the plain sum (strict generalization of v4).

Representation-level contrastive regularization (cl_lambda > 0), computed
on the unique doc-final nodes of each batch:

- `cl_mode: "supcon"` (default, paper L_CL): label-aware weighted SupCon.
  Positives share the stance label; a pair that also shares the target gets
  weight cl_pos_weight_same_target, a same-stance cross-target pair gets
  cl_pos_weight_diff_target; different-stance anchors are negatives.
- `cl_mode: "augment"` (legacy, the 0.4536 configuration): self-supervised
  InfoNCE between two corrupted views (feature masking + Gaussian noise)
  of the same hidden state via the shared projection head.

CE is computed on the clean `h` so supervised gradients stay unpolluted,
evaluation is deterministic (no augmentation, no contrastive loss).

(A graph-level variant -- perturbing input features, pair edges and
hyperedge memberships with a second full encode -- was evaluated and HURT:
dev f1_macro 0.4341 vs 0.4510 baseline / 0.4536 repr. It was removed; do
not re-add without a new hypothesis.)

Target-side stance semantic grounding (use_grounding=True): a frozen prior
([G, 4, D] per target: raw target text + background/favor/against views,
encoded offline by the SAME frozen PLM) is attached to every batch. The
trace h_i^tr attends over the three stance views (TargetGrounding) to
produce h_i^tar, and the classifier consumes h_i = [h_i^tr; h_i^tar; t_i].

Prediction protocol unchanged from v3/v4: every node is predicted; training
loss is CE with ignore_index=-100.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HypergraphConv, RGCNConv

IGNORE_LABEL = -100

NUM_CHANNELS = 4  # spk / dlg / temporal / reply


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE between two L2-normalized view matrices [N, D].

    Row i of z1 is matched with row i of z2 (positive pair); all other rows
    act as in-batch negatives for both directions.
    """
    n = z1.size(0)
    if n < 2:
        return z1.new_zeros(())
    logits = z1 @ z2.t() / max(1e-6, float(temperature))
    targets = torch.arange(n, device=z1.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.t(), targets)
    )


def _supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    same_group: torch.Tensor,
    temperature: float,
    w_same_target: float,
    w_diff_target: float,
) -> torch.Tensor:
    """Weighted supervised contrastive loss (SupCon) over anchor samples.

    Positives are anchors sharing the stance label. A positive pair that
    also shares the target gets weight `w_same_target`; a same-stance
    cross-target pair gets `w_diff_target` (paper L_CL). The log-prob
    denominator spans ALL other anchors, so different-stance anchors act as
    negatives. Anchors with IGNORE_LABEL or without any positive are
    skipped.
    """
    valid = labels != IGNORE_LABEL
    if not bool(valid.any()):
        return z.new_zeros(())
    z = z[valid]
    labels = labels[valid]
    same_group = same_group[valid][:, valid]
    m = z.size(0)
    eye = torch.eye(m, dtype=torch.bool, device=z.device)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye  # [M, M]
    weights = torch.where(
        same_group,
        z.new_full((), float(w_same_target)),
        z.new_full((), float(w_diff_target)),
    ) * pos
    w_sum = weights.sum(dim=1)
    active = w_sum > 0
    if not active.any():
        return z.new_zeros(())

    sim = z @ z.t() / max(1e-6, float(temperature))
    # denominator excludes the self-pair only; self-similarity never enters
    # the numerator (weights are zero on the diagonal), so no inf/NaN paths.
    log_prob = sim - torch.logsumexp(
        sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True
    )
    per_anchor = (weights * log_prob).sum(dim=1)[active] / w_sum[active]
    return -per_anchor.mean()


class TargetGrounding(nn.Module):
    """Ground the stance trace to target semantics via stance-view attention.

    The frozen prior ([G, V, D], V = 3 stance views k^B / k^F / k^A) is
    projected once per batch; every trace node h_i^tr attends over its own
    graph's views:

        alpha_i = softmax((W_q h_i)(W_k K)^T / sqrt(d))
        h_i^tar = W_o (alpha_i W_v K)

    The raw target text row t_i of the prior is NOT attended over; it is
    concatenated unchanged into the final representation by the model
    (h_i = [h_i^tr; h_i^tar; t_i]). Because K comes from the SAME frozen
    encoder as the utterance features, query and keys start in one
    representation space; W_q/W_k/W_v/W_o carry all learnable adaptation.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.scale = hidden_size ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        h: torch.Tensor,
        knowledge: torch.Tensor,
        graph_ptr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: [N, D] trace representations; knowledge: [G, V, D] frozen prior
        (V = 3 stance views B/F/A; the raw target row is handled outside);
        graph_ptr: [G+1] node offsets. Returns (h_tar [N, D], alpha [N, V])."""
        n, d = h.shape
        num_graphs = knowledge.size(0)
        if num_graphs == 0 or n == 0:
            return torch.zeros_like(h), h.new_zeros((n, knowledge.size(1)))

        # per-node graph id from the batch offsets, then gather each node's
        # own [V, D] prior: k_node [N, V, D]
        node_graph = torch.repeat_interleave(
            torch.arange(num_graphs, device=h.device),
            (graph_ptr[1:] - graph_ptr[:-1]).clamp_min(0),
        )
        k_node = knowledge[node_graph]  # [N, V, D]

        q = self.q_proj(h)  # [N, D]
        k = self.k_proj(k_node)  # [N, V, D]
        v = self.v_proj(k_node)  # [N, V, D]

        # alpha_i over the V views: [N, V]
        alpha = torch.softmax(
            torch.einsum("nd,nvd->nv", q, k) * self.scale, dim=-1
        )
        grounded = torch.einsum("nv,nvd->nd", alpha, v)  # [N, D]
        grounded = self.out_proj(self.dropout(grounded))
        return self.norm(grounded), alpha


class GatedFusion(nn.Module):
    """Per-node content-based gate over K relation messages.

    gates = softmax(Gate([h, sum_i m_i])); fused = K_active * sum_i g_i * m_i.
    Zero-initialized gate head => init gates = 1/K_active over the ACTIVE
    channels => fused = sum of active messages (the plain-sum fusion), so
    this is a strict generalization rather than a different starting point.

    Ablation support: pass ``active`` (bools over the 4 channels). Logits
    of inactive channels are masked with -inf BEFORE softmax, so removed
    channels consume zero gate mass and the rescale uses K_active instead
    of K. Without ``active`` (or all True) behaviour is identical to v5.
    """

    def __init__(self, hidden_size: int, num_channels: int = NUM_CHANNELS):
        super().__init__()
        self.num_channels = int(num_channels)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_channels),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        messages: List[torch.Tensor],
        active: Optional[List[bool]] = None,
        return_gate: bool = False,
    ):
        assert len(messages) == self.num_channels
        if active is None:
            active = [True] * self.num_channels
        assert len(active) == self.num_channels
        num_active = int(sum(1 for a in active if a))
        gate = torch.zeros((h.size(0), self.num_channels), device=h.device)
        if num_active == 0:
            return (torch.zeros_like(h), gate) if return_gate else torch.zeros_like(h)
        evidence = torch.zeros_like(h)
        for m, a in zip(messages, active):
            if a:
                evidence = evidence + m
        logits = self.gate(torch.cat([h, evidence], dim=-1))  # [N, K]
        mask = torch.tensor(active, dtype=torch.bool, device=h.device).view(1, -1)
        logits = logits.masked_fill(~mask, float("-inf"))
        gate = F.softmax(logits, dim=-1)  # [N, K], inactive cols are 0
        fused = torch.zeros_like(h)
        for i, m in enumerate(messages):
            if active[i]:
                fused = fused + gate[:, i : i + 1] * m
        fused = float(num_active) * fused
        if return_gate:
            return fused, gate
        return fused


class HeterogeneousStanceTracerV5(nn.Module):
    """PyG-based stance tracer with FOUR independent fusion channels.

    Channels (each its own conv stack, like the hyperedge families):

        0. speaker hypergraph   -> m_spk
        1. dialogue hypergraph  -> m_dlg
        2. temporal graph       -> m_temp  (RGCNConv, relation 0 only)
        3. reply-candidate graph-> m_reply (RGCNConv, relation 1 only)

    Splitting the single v5 RGCNConv(num_relations=2) into two
    RGCNConv(num_relations=1) stacks gives every binary relation its own
    message, so the fusion gate can weight them separately and the case
    study can read per-relation attention.

    Component ablation via config flags (all default True = full model):

      - ``use_speaker``   : speaker hypergraph channel (m_spk)
      - ``use_dialogue``  : dialogue hypergraph channel (m_dlg)
      - ``use_temporal``  : temporal channel (m_temp)
      - ``use_reply``     : reply-candidate channel (m_reply)

    Disabled channels are zeroed BEFORE fusion and their gate logits are
    masked with -inf, so the remaining channels renormalize to full gate
    mass (the fusion rescales by K_active).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = int(getattr(config, "hidden_size", 768))
        self.num_classes = int(getattr(config, "num_classes", 3))
        self.dropout = float(getattr(config, "dropout", 0.1))
        self.input_dropout = float(getattr(config, "input_dropout", 0.0))
        self.num_layers = max(1, int(getattr(config, "num_layers", 1)))
        num_bases = int(getattr(config, "rgcn_num_bases", 0)) or None
        self.hypergraph_attention = bool(getattr(config, "hypergraph_attention", True))

        # ---- ablation switches (True = keep channel) ----
        self.use_speaker = bool(getattr(config, "use_speaker", True))
        self.use_dialogue = bool(getattr(config, "use_dialogue", True))
        self.use_temporal = bool(getattr(config, "use_temporal", True))
        self.use_reply = bool(getattr(config, "use_reply", True))

        self.input_size = int(getattr(config, "input_size", 0) or self.hidden_size)
        self.input_proj = (
            nn.Identity() if self.input_size == self.hidden_size else
            nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.LayerNorm(self.hidden_size))
        )

        self.speaker_convs = nn.ModuleList(
            [
                HypergraphConv(
                    self.hidden_size,
                    self.hidden_size,
                    use_attention=self.hypergraph_attention,
                    heads=1,
                    concat=False,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.dialogue_convs = nn.ModuleList(
            [
                HypergraphConv(
                    self.hidden_size,
                    self.hidden_size,
                    use_attention=self.hypergraph_attention,
                    heads=1,
                    concat=False,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.graph_convs = nn.ModuleList(
            [
                RGCNConv(
                    self.hidden_size,
                    self.hidden_size,
                    num_relations=1,  # temporal edges only
                    num_bases=num_bases,
                    root_weight=True,
                    bias=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.reply_convs = nn.ModuleList(
            [
                RGCNConv(
                    self.hidden_size,
                    self.hidden_size,
                    num_relations=1,  # reply-candidate edges only
                    num_bases=num_bases,
                    root_weight=True,
                    bias=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)]
        )
        self.fuse_gates = nn.ModuleList(
            [GatedFusion(self.hidden_size, num_channels=NUM_CHANNELS) for _ in range(self.num_layers)]
        )

        # Target-side stance semantic grounding (disabled until a knowledge
        # prior of the right width arrives in the batch).
        self.use_grounding = bool(getattr(config, "use_grounding", False))
        # Off by default: preserve all existing experiments/checkpoints.
        self.use_target_claim_mlp = bool(getattr(config, "use_target_claim_mlp", False))
        # Residual claim head: claim stance often opposes a related noun target.
        self.use_claim_noun_polarity = bool(getattr(config, "use_claim_noun_polarity", False))
        if self.use_target_claim_mlp and not self.use_grounding:
            raise ValueError("use_target_claim_mlp requires use_grounding=True and target priors")
        if self.use_grounding:
            self.grounding = TargetGrounding(self.hidden_size, dropout=self.dropout)
            # h: contextual node/claim; t: raw target; h_tar: stance-view grounding.
            # Difference is directional, while product models alignment.
            if self.use_target_claim_mlp:
                pair_dim = self.hidden_size * 5 + 1
                self.pair_mlp = nn.Sequential(
                    nn.LayerNorm(pair_dim),
                    nn.Linear(pair_dim, self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                )
                self.pair_gate = nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid()
                )
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_classes),
            )
        else:
            self.grounding = None
            self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        if self.use_claim_noun_polarity:
            # [h, t_claim, t_noun, sim, negated] -> 3-class residual, zero-init
            # so the first step matches the original classifier.
            self.claim_adj = nn.Sequential(
                nn.Linear(self.hidden_size * 3 + 2, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.num_classes),
            )
            nn.init.zeros_(self.claim_adj[-1].weight)
            nn.init.zeros_(self.claim_adj[-1].bias)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

        # Representation-level contrastive regularization
        self.cl_lambda = float(getattr(config, "cl_lambda", 0.0))
        self.cl_temperature = float(getattr(config, "cl_temperature", 0.2))
        # "supcon": label-aware weighted SupCon (paper L_CL, default);
        # "augment": legacy self-supervised InfoNCE between two corrupted views.
        self.cl_mode = str(getattr(config, "cl_mode", "supcon")).lower()
        if self.cl_mode not in ("supcon", "augment"):
            self.cl_mode = "supcon"
        self.cl_pos_weight_same_target = float(getattr(config, "cl_pos_weight_same_target", 1.0))
        self.cl_pos_weight_diff_target = float(getattr(config, "cl_pos_weight_diff_target", 0.5))
        self.cl_feat_mask_rate = float(getattr(config, "cl_feat_mask_rate", 0.2))
        self.cl_noise_scale = float(getattr(config, "cl_noise_scale", 0.1))
        proj_dim = int(getattr(config, "cl_proj_dim", 128))
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, proj_dim),
        )

    def _temporal_message(self, i: int, h: torch.Tensor, temporal_edge_index: torch.Tensor) -> torch.Tensor:
        """Temporal-chain message from its OWN RGCNConv (temporal only).

        The conv has num_relations=1, so its local relation id is 0; the
        temporal/reply distinction lives in the separate weight stacks,
        not in edge_type.
        """
        if not self.use_temporal or temporal_edge_index.size(1) == 0:
            return torch.zeros_like(h)
        edge_type = torch.zeros(temporal_edge_index.size(1), dtype=torch.long, device=h.device)
        return self.graph_convs[i](h, temporal_edge_index, edge_type)

    def _reply_message(self, i: int, h: torch.Tensor, reply_candidate_edge_index: torch.Tensor) -> torch.Tensor:
        """Reply-candidate message from its OWN RGCNConv (reply only).

        RGCNConv (PyG 2.6) aggregates with `aggr='mean'` per receiving node,
        which keeps the dense reply-candidate neighborhoods at the same
        magnitude as the temporal chain without any manual edge weighting.
        num_relations=1 again, so the local relation id is 0.
        """
        if not self.use_reply or reply_candidate_edge_index.size(1) == 0:
            return torch.zeros_like(h)
        edge_type = torch.zeros(reply_candidate_edge_index.size(1), dtype=torch.long, device=h.device)
        return self.reply_convs[i](h, reply_candidate_edge_index, edge_type)

    def _hyperedge_attr(
        self, h: torch.Tensor, hyperedge_index: torch.Tensor
    ) -> torch.Tensor:
        """Per-hyperedge descriptor required by HypergraphConv attention.

        PyG's attention variant computes node-hyperedge attention against a
        `[M, F]` hyperedge feature matrix; we use the degree-normalized mean
        of the members' current representations (no extra parameters, no
        trainable per-hyperedge embeddings that could leak target identity).
        """
        num_edges = int(hyperedge_index[1].max().item()) + 1
        members, e_ids = hyperedge_index[0], hyperedge_index[1]
        deg = torch.zeros(num_edges, dtype=h.dtype, device=h.device)
        deg.index_add_(0, e_ids, torch.ones_like(e_ids, dtype=h.dtype))
        summ = torch.zeros(num_edges, h.size(1), dtype=h.dtype, device=h.device)
        summ.index_add_(0, e_ids, h[members])
        return summ / deg.clamp_min(1.0).unsqueeze(-1)

    def _encode(
        self,
        x: torch.Tensor,
        speaker_hyperedge_index: torch.Tensor,
        dialogue_hyperedge_index: torch.Tensor,
        temporal_edge_index: torch.Tensor,
        reply_candidate_edge_index: torch.Tensor,
    ):
        """Run the stacked conv blocks.

        Returns (h, m_spk, m_dlg, m_temp, m_reply, last_gate) where
        last_gate is the final layer's per-node gate weight [N, 4] over
        [spk, dlg, temporal, reply] -- the quantity the case study reads.

        Ablated channels emit zeros and receive zero gate mass (their gate
        logits are masked before softmax), so the active channels fully
        determine the fusion.
        """
        h = x
        m_spk = m_dlg = m_temp = m_reply = torch.zeros_like(x)
        active = [self.use_speaker, self.use_dialogue, self.use_temporal, self.use_reply]
        last_gate: Optional[torch.Tensor] = None
        for i in range(self.num_layers):
            if self.use_speaker and speaker_hyperedge_index.size(1) > 0:
                m_spk = self.speaker_convs[i](
                    h,
                    speaker_hyperedge_index,
                    hyperedge_attr=self._hyperedge_attr(h, speaker_hyperedge_index),
                )
            else:
                m_spk = torch.zeros_like(h)
            if self.use_dialogue and dialogue_hyperedge_index.size(1) > 0:
                m_dlg = self.dialogue_convs[i](
                    h,
                    dialogue_hyperedge_index,
                    hyperedge_attr=self._hyperedge_attr(h, dialogue_hyperedge_index),
                )
            else:
                m_dlg = torch.zeros_like(h)
            m_temp = self._temporal_message(i, h, temporal_edge_index)
            m_reply = self._reply_message(i, h, reply_candidate_edge_index)
            fused, gate = self.fuse_gates[i](
                h, [m_spk, m_dlg, m_temp, m_reply], active=active, return_gate=True
            )
            h = self.layer_norms[i](h + fused)
            last_gate = gate
        return h, m_spk, m_dlg, m_temp, m_reply, last_gate

    def _perturb_h(self, h: torch.Tensor) -> torch.Tensor:
        """Representation-level corruption: feature masking + Gaussian noise.

        Two independent calls yield two views of the same hidden state. The
        masking zeros out dims with rate cl_feat_mask_rate (Bernoulli per
        dim per node), then N(0, cl_noise_scale) noise is added.
        """
        v = h
        if self.cl_feat_mask_rate > 0.0:
            keep = (torch.rand_like(v) >= self.cl_feat_mask_rate).to(v.dtype)
            v = v * keep
        if self.cl_noise_scale > 0.0:
            v = v + torch.randn_like(v) * self.cl_noise_scale
        return v

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        speaker_hyperedge_index: Optional[torch.Tensor] = None,
        dialogue_hyperedge_index: Optional[torch.Tensor] = None,
        temporal_edge_index: Optional[torch.Tensor] = None,
        reply_candidate_edge_index: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,  # dialogue_ids / utterance_ids / doc_final_* / graph_ptr ... metadata
    ) -> Dict[str, Any]:
        device = x.device
        x = self.input_proj(x)
        if self.input_dropout > 0:
            x = F.dropout(x, p=self.input_dropout, training=self.training)

        if speaker_hyperedge_index is None:
            speaker_hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        if dialogue_hyperedge_index is None:
            dialogue_hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        if temporal_edge_index is None:
            temporal_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        if reply_candidate_edge_index is None:
            reply_candidate_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        h, m_spk, m_dlg, m_temp, m_reply, fusion_gate = self._encode(
            x,
            speaker_hyperedge_index,
            dialogue_hyperedge_index,
            temporal_edge_index,
            reply_candidate_edge_index,
        )

        h_tar = alpha = None
        target_knowledge = kwargs.get("target_knowledge", None)
        graph_ptr = kwargs.get("graph_ptr", None)
        if self.use_grounding:
            if (
                target_knowledge is None
                or target_knowledge.dim() != 3
                or target_knowledge.size(0) == 0
                or graph_ptr is None
                or graph_ptr.numel() != target_knowledge.size(0) + 1
            ):
                raise RuntimeError(
                    "use_grounding=True but the batch lacks a valid "
                    "target_knowledge prior (or graph_ptr mismatch); build the "
                    "datasets with the same config "
                    "(run build_target_embeddings.py --stage targets first)"
                )
            # row 0 of the prior is the raw target text t_i (un-grounded
            # target representation); rows 1..3 are the stance views K.
            target_knowledge = self.input_proj(target_knowledge)
            t_repr = target_knowledge[:, 0, :]  # [G, D]
            k_views = target_knowledge[:, 1:, :]  # [G, V, D]
            h_tar, alpha = self.grounding(h, k_views, graph_ptr)
            g_idx = torch.repeat_interleave(
                torch.arange(target_knowledge.size(0), device=device),
                (graph_ptr[1:] - graph_ptr[:-1]).clamp_min(0),
            )
            t_nodes = t_repr[g_idx]
            if self.use_target_claim_mlp:
                # All features are target-conditioned. A negative similarity alone
                # is NOT treated as an Against label: the head learns from gold labels.
                similarity = F.cosine_similarity(h, t_nodes, dim=-1).unsqueeze(-1)
                pair_features = torch.cat(
                    [h, h_tar, t_nodes, h - t_nodes, h * t_nodes, similarity], dim=-1
                )
                pair_repr = self.pair_mlp(pair_features)
                gate = self.pair_gate(torch.cat([h, pair_repr], dim=-1))
                # Residual leaves existing graph context directly accessible.
                logits = self.classifier(torch.cat([h + gate * pair_repr, h_tar, t_nodes], dim=-1))
            else:
                logits = self.classifier(torch.cat([h, h_tar, t_nodes], dim=-1))
            if self.use_claim_noun_polarity:
                type_ids = kwargs.get("target_type_ids", None)
                noun_know = kwargs.get("related_noun_knowledge", None)
                claim_neg = kwargs.get("claim_negated", None)
                if (
                    type_ids is not None
                    and noun_know is not None
                    and claim_neg is not None
                    and noun_know.dim() == 3
                    and noun_know.size(1) == 4
                    and noun_know.size(-1) > 0
                ):
                    noun_know = self.input_proj(noun_know)
                    t_noun = noun_know[:, 0, :][g_idx]
                    sim_noun = F.cosine_similarity(h, t_noun, dim=-1).unsqueeze(-1)
                    neg = claim_neg.to(device=device, dtype=h.dtype).unsqueeze(-1)
                    neg_nodes = neg[g_idx]
                    adj = self.claim_adj(
                        torch.cat([h, t_nodes, t_noun, sim_noun, neg_nodes], dim=-1)
                    )
                    claim_mask = (type_ids.view(-1).to(device) == 1).unsqueeze(-1).to(adj.dtype)
                    logits = logits + claim_mask * adj
        else:
            logits = self.classifier(h)  # [N_total_nodes, num_classes]

        loss = None
        if y is not None:
            y = y.view(-1).long().to(logits.device)
            if (y != IGNORE_LABEL).any():
                loss = self.criterion(logits, y)
            else:
                loss = logits.sum() * 0.0  # keep the graph in autograd

        loss_cl = None
        doc_final_node_index = kwargs.get("doc_final_node_index", None)
        if (
            self.training
            and self.cl_lambda > 0.0
            and y is not None
            and doc_final_node_index is not None
            and doc_final_node_index.numel() >= 2
        ):
            # Unique doc-final nodes: a shared sentence that ends several docs
            # would otherwise appear twice with identical h, yielding a
            # trivial +1 similarity pair.
            anchors = torch.unique(doc_final_node_index.view(-1).long())
            if self.cl_mode == "augment":
                h1 = self._perturb_h(h[anchors])
                h2 = self._perturb_h(h[anchors])
                z1 = F.normalize(self.proj_head(h1), dim=-1)
                z2 = F.normalize(self.proj_head(h2), dim=-1)
                loss_cl = _info_nce(z1, z2, self.cl_temperature)
            else:
                y_anchors = y[anchors]
                target_ids = kwargs.get("target_ids", None)
                if target_ids is None:
                    raise RuntimeError(
                        "cl_mode='supcon' requires target_ids in the batch "
                        "(the collate function must keep graph-level target ids)"
                    )
                group = target_ids.view(-1).long()[anchors]
                z = F.normalize(self.proj_head(h[anchors]), dim=-1)
                same_group = group.view(-1, 1) == group.view(1, -1)
                loss_cl = _supcon_loss(
                    z=z,
                    labels=y_anchors.long(),
                    same_group=same_group,
                    temperature=self.cl_temperature,
                    w_same_target=self.cl_pos_weight_same_target,
                    w_diff_target=self.cl_pos_weight_diff_target,
                )
            loss = loss + self.cl_lambda * loss_cl

        output = {
            "loss": loss,
            "loss_contrastive": loss_cl.detach() if loss_cl is not None else None,
            "logits": logits,
            "labels": y if y is not None else None,
            # channel messages (4 independent channels for the case study)
            "h_spk": m_spk,
            "h_dlg": m_dlg,
            "h_temp": m_temp,
            "h_reply": m_reply,
            # legacy alias: the combined binary-relation evidence
            "h_graph": m_temp + m_reply,
            # per-node fusion gate [N, 4] over [spk, dlg, temporal, reply];
            # rows sum to 1 over the ACTIVE channels (0 for ablated ones)
            "fusion_gate": fusion_gate,
            "h_trace": h,
            "h_target": h_tar,
            "target_alpha": alpha,
        }
        if return_dict:
            return output
        return logits
