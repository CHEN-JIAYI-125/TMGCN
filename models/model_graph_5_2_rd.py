#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HeterogeneousStanceTracerV5 + Reply Structure Knowledge Distillation。

本文件 **继承** ``models.model_graph_5_2.HeterogeneousStanceTracerV5``，不删除
也不改写 Speaker / Dialogue 超图、Temporal RGCN、Gated Fusion、Target
Grounding、Contrastive Learning。Reply Distillation 只替换回复通道：

    冻结句向量 x
        → Student ReplyPredictor → P_S(j|t)
        → （可选）与 LLM Teacher P_T 做 KL 蒸馏
        → 阈值化得到 soft_reply_edge_index / edge_weight
        → Reply RGCN(node, edge_index, edge_weight)
        → 与其它三路一起 gated fusion

开关 ``reply_distillation=False``（默认）时：
    - 不创建 Student 参数；
    - ``forward`` 直接调用父类，回复边仍是等权 candidate 图；
    - 数值行为与原 TMGCN baseline 完全等价。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from models.model_graph_5_2 import (
    IGNORE_LABEL,
    HeterogeneousStanceTracerV5,
    _info_nce,
    _supcon_loss,
)
from modules.reply_distillation import (
    ReplyKDLoss,
    ReplyPredictor,
    build_soft_reply_graph,
    merge_teacher_fallback_edges,
)


def _rel0_linear(conv: Any, x: torch.Tensor) -> torch.Tensor:
    """取出 RGCNConv 第 0 个关系的线性变换，兼容 PyG 2.3–2.6。"""
    weight = getattr(conv, "weight", None)
    if weight is not None:
        w = weight[0] if weight.dim() == 3 else weight
        return x.matmul(w)
    for name in ("lin", "comp"):
        lin = getattr(conv, name, None)
        if lin is not None and callable(lin):
            return lin(x)
    raise RuntimeError("当前 RGCNConv 没有可用的关系权重，无法做加权聚合")


def _root_linear(conv: Any, x: torch.Tensor) -> torch.Tensor:
    root = getattr(conv, "root", None)
    if root is not None:
        if root.dim() == 2:
            return x.matmul(root)
        return x + root
    lin_root = getattr(conv, "lin_root", None)
    if lin_root is not None:
        return lin_root(x)
    return torch.zeros_like(x)


def _weighted_rgcn_mean(
    conv: Any,
    h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """旧版 RGCNConv 不接受 edge_weight 时：message = w_e * (x_j @ W)，再按入度取均值。

    与 PyG 2.6 在 ``aggr='mean'`` 且传入 ``edge_weight`` 时的行为一致。
    """
    src, dst = edge_index[0], edge_index[1]
    msg = _rel0_linear(conv, h[src]) * edge_weight.unsqueeze(-1)
    out = torch.zeros(h.size(0), msg.size(-1), device=h.device, dtype=h.dtype)
    out.index_add_(0, dst, msg)
    deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
    deg.index_add_(0, dst, torch.ones_like(edge_weight))
    out = out / deg.clamp_min(1.0).unsqueeze(-1)
    out = out + _root_linear(conv, h)
    bias = getattr(conv, "bias", None)
    if bias is not None:
        out = out + bias
    return out


class HeterogeneousStanceTracerV5RD(HeterogeneousStanceTracerV5):
    """在 v5 四通道融合之上，把 Reply 通道换成概率回复图。"""

    def __init__(self, config):
        super().__init__(config)
        # 两个名字是同一总开关：yaml 里写 reply_distillation 或 use_reply_distillation 均可
        self.reply_distillation = bool(
            getattr(config, "reply_distillation", False)
            or getattr(config, "use_reply_distillation", False)
        )
        self.use_reply_kd = bool(getattr(config, "use_reply_kd", True))
        self.use_soft_reply_graph = bool(getattr(config, "use_soft_reply_graph", True))
        self.reply_graph_source = str(getattr(config, "reply_graph_source", "student")).lower()
        if self.reply_graph_source not in ("student", "teacher", "candidate"):
            self.reply_graph_source = "student"
        self.reply_kd_lambda = float(getattr(config, "reply_kd_lambda", 0.5))
        self.reply_threshold = float(getattr(config, "reply_threshold", 0.05))
        # 沿用原 GraphConfig.reply_temperature；蒸馏默认建议 3
        self.reply_kd_temperature = float(getattr(config, "reply_temperature", 3.0))

        self.reply_predictor: Optional[ReplyPredictor] = None
        self.reply_kd_criterion: Optional[ReplyKDLoss] = None
        need_student = self.reply_distillation and (
            self.use_reply_kd
            or (self.use_soft_reply_graph and self.reply_graph_source == "student")
        )
        if need_student:
            self.reply_predictor = ReplyPredictor(
                hidden_size=self.hidden_size,
                dropout=self.dropout,
            )
        if self.reply_distillation and self.use_reply_kd:
            self.reply_kd_criterion = ReplyKDLoss(temperature=self.reply_kd_temperature)

    # ------------------------------------------------------------------
    # Reply RGCN：在父类基础上增加 edge_weight
    # ------------------------------------------------------------------
    def _reply_message(
        self,
        i: int,
        h: torch.Tensor,
        reply_candidate_edge_index: torch.Tensor,
        reply_edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reply 通道消息。关闭蒸馏时不传 edge_weight，与父类逐算子等价。

        开启 Soft Reply Graph 时，``edge_weight`` 为 ``P(j|t)``，PyG RGCNConv
        会在 message 里把 ``x_j`` 乘上该权重，从而让高回复概率的 antecedent
        对当前话语贡献更大。
        """
        if not self.use_reply or reply_candidate_edge_index.size(1) == 0:
            return torch.zeros_like(h)
        edge_type = torch.zeros(
            reply_candidate_edge_index.size(1), dtype=torch.long, device=h.device
        )
        conv = self.reply_convs[i]
        if reply_edge_weight is None:
            return conv(h, reply_candidate_edge_index, edge_type)
        weight = reply_edge_weight.to(device=h.device, dtype=h.dtype).reshape(-1)
        if weight.numel() != reply_candidate_edge_index.size(1):
            raise RuntimeError(
                "reply edge_weight 长度 (%d) 与边数 (%d) 不一致"
                % (int(weight.numel()), int(reply_candidate_edge_index.size(1)))
            )
        # 新版 PyG 的 RGCNConv 支持 edge_weight；服务器上的旧版没有这个参数。
        if "edge_weight" in inspect.signature(conv.forward).parameters:
            return conv(h, reply_candidate_edge_index, edge_type, edge_weight=weight)
        return _weighted_rgcn_mean(conv, h, reply_candidate_edge_index, weight)

    def _encode(
        self,
        x: torch.Tensor,
        speaker_hyperedge_index: torch.Tensor,
        dialogue_hyperedge_index: torch.Tensor,
        temporal_edge_index: torch.Tensor,
        reply_candidate_edge_index: torch.Tensor,
        reply_edge_weight: Optional[torch.Tensor] = None,
    ):
        """与父类相同的四通道堆叠，仅 Reply 通道多了 ``reply_edge_weight``。"""
        h = x
        m_spk = m_dlg = m_temp = m_reply = torch.zeros_like(x)
        active = [self.use_speaker, self.use_dialogue, self.use_temporal, self.use_reply]
        last_gate = None
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
            m_reply = self._reply_message(
                i, h, reply_candidate_edge_index, reply_edge_weight=reply_edge_weight
            )
            fused, gate = self.fuse_gates[i](
                h, [m_spk, m_dlg, m_temp, m_reply], active=active, return_gate=True
            )
            h = self.layer_norms[i](h + fused)
            last_gate = gate
        return h, m_spk, m_dlg, m_temp, m_reply, last_gate

    def _build_reply_graph_from_probs(
        self,
        probs: torch.Tensor,
        valid_hist: Optional[torch.Tensor],
        candidate_edge_index: torch.Tensor,
        teacher_row_valid: Optional[torch.Tensor] = None,
        fallback_empty: bool = False,
    ):
        """阈值化概率分布，得到 Reply RGCN 的边与边权。"""
        edge_index, edge_weight = build_soft_reply_graph(
            probs, threshold=self.reply_threshold, valid_hist=valid_hist
        )
        if fallback_empty and teacher_row_valid is not None:
            edge_index, edge_weight = merge_teacher_fallback_edges(
                edge_index, edge_weight, candidate_edge_index, teacher_row_valid
            )
        return edge_index, edge_weight

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        speaker_hyperedge_index: Optional[torch.Tensor] = None,
        dialogue_hyperedge_index: Optional[torch.Tensor] = None,
        temporal_edge_index: Optional[torch.Tensor] = None,
        reply_candidate_edge_index: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        # ---- 总开关关闭：父类前向，结果与 baseline 逐张量等价 ----
        if not self.reply_distillation:
            output = super().forward(
                x=x,
                y=y,
                speaker_hyperedge_index=speaker_hyperedge_index,
                dialogue_hyperedge_index=dialogue_hyperedge_index,
                temporal_edge_index=temporal_edge_index,
                reply_candidate_edge_index=reply_candidate_edge_index,
                return_dict=True,
                **kwargs,
            )
            output["stance_loss"] = output.get("loss")
            output["reply_kd_loss"] = None
            output["total_loss"] = output.get("loss")
            if not return_dict:
                return output["logits"]
            return output

        device = x.device
        x = self.input_proj(x)
        # Student 必须看 dropout 之前的冻结句向量，图结构才稳定，
        # 也避免 Reply RGCN 的输入与 Student 输入互相依赖。
        student_in = x
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

        teacher_probs = kwargs.get("reply_teacher_distribution", None)
        teacher_mask = kwargs.get("reply_teacher_mask", None)

        reply_edge_index = reply_candidate_edge_index
        reply_edge_weight = None
        p_s = student_logits = valid_hist = None
        loss_reply_kd = None

        if self.reply_predictor is not None:
            p_s, student_logits, valid_hist = self.reply_predictor(
                student_in, candidate_edge_index=reply_candidate_edge_index
            )

        # ---- 构造 Reply 通道使用的图 ----
        if self.use_soft_reply_graph and self.reply_graph_source == "student" and p_s is not None:
            reply_edge_index, reply_edge_weight = self._build_reply_graph_from_probs(
                p_s, valid_hist, reply_candidate_edge_index
            )
        elif self.use_soft_reply_graph and self.reply_graph_source == "teacher":
            if teacher_probs is not None and teacher_mask is not None:
                # 只在 Teacher support 上保留概率，其它位置视为 0
                p_t_graph = teacher_probs.to(device=device, dtype=student_in.dtype)
                t_mask = teacher_mask.to(device=device, dtype=torch.bool)
                p_t_graph = p_t_graph * t_mask.to(p_t_graph.dtype)
                row_z = p_t_graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                p_t_graph = p_t_graph / row_z
                hist_valid = t_mask[:, 1:]
                row_valid = t_mask.any(dim=-1)
                reply_edge_index, reply_edge_weight = self._build_reply_graph_from_probs(
                    p_t_graph,
                    hist_valid,
                    reply_candidate_edge_index,
                    teacher_row_valid=row_valid,
                    fallback_empty=True,
                )
            # teacher 文件完全缺失时保持原 candidate 图（等权）
        # source == "candidate" 或 use_soft_reply_graph=False：沿用原图，不传 weight

        h, m_spk, m_dlg, m_temp, m_reply, fusion_gate = self._encode(
            x,
            speaker_hyperedge_index,
            dialogue_hyperedge_index,
            temporal_edge_index,
            reply_edge_index,
            reply_edge_weight=reply_edge_weight,
        )

        # ---- Target Grounding + 分类头：与父类完全相同 ----
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
            target_knowledge = self.input_proj(target_knowledge)
            t_repr = target_knowledge[:, 0, :]
            k_views = target_knowledge[:, 1:, :]
            h_tar, alpha = self.grounding(h, k_views, graph_ptr)
            g_idx = torch.repeat_interleave(
                torch.arange(target_knowledge.size(0), device=device),
                (graph_ptr[1:] - graph_ptr[:-1]).clamp_min(0),
            )
            logits = self.classifier(torch.cat([h, h_tar, t_repr[g_idx]], dim=-1))
        else:
            logits = self.classifier(h)

        loss_stance = None
        if y is not None:
            y = y.view(-1).long().to(logits.device)
            if (y != IGNORE_LABEL).any():
                loss_stance = self.criterion(logits, y)
            else:
                loss_stance = logits.sum() * 0.0  # keep the graph in autograd

        loss = loss_stance
        loss_cl = None
        doc_final_node_index = kwargs.get("doc_final_node_index", None)
        if (
            self.training
            and self.cl_lambda > 0.0
            and y is not None
            and doc_final_node_index is not None
            and doc_final_node_index.numel() >= 2
        ):
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

        # ---- Reply KD：只在有 Teacher 的节点上计算 ----
        if (
            self.training
            and self.use_reply_kd
            and self.reply_kd_criterion is not None
            and student_logits is not None
            and teacher_probs is not None
            and teacher_mask is not None
        ):
            loss_reply_kd = self.reply_kd_criterion(
                student_logits,
                teacher_probs.to(device=device, dtype=student_logits.dtype),
                teacher_mask.to(device=device),
            )
            loss = loss + self.reply_kd_lambda * loss_reply_kd

        output = {
            "loss": loss,
            "stance_loss": loss_stance.detach() if loss_stance is not None else None,
            "reply_kd_loss": loss_reply_kd.detach() if loss_reply_kd is not None else None,
            "total_loss": loss.detach() if loss is not None else None,
            "loss_contrastive": loss_cl.detach() if loss_cl is not None else None,
            "logits": logits,
            "labels": y if y is not None else None,
            "h_spk": m_spk,
            "h_dlg": m_dlg,
            "h_temp": m_temp,
            "h_reply": m_reply,
            "h_graph": m_temp + m_reply,
            "fusion_gate": fusion_gate,
            "h_trace": h,
            "h_target": h_tar,
            "target_alpha": alpha,
            "reply_probs": p_s.detach() if p_s is not None else None,
            "soft_reply_edge_index": reply_edge_index.detach() if reply_edge_index is not None else None,
            "soft_reply_edge_weight": (
                reply_edge_weight.detach() if reply_edge_weight is not None else None
            ),
        }
        if return_dict:
            return output
        return logits
