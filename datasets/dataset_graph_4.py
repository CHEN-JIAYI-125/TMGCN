#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/6/8

"""PyG-standard target-level graph dataset (v4: two hyperedge families).

Same unique-sentence graph construction as dataset_graph_3 (one node per
global utterance id per target; first-occurrence label/speaker; temporal and
reply edges unioned across a sentence's occurrences), but the topologies are
exposed in PyG-standard sparse formats so PyG's own conv operators can be
used directly:

  - `speaker_hyperedge_index`   [2, E_s]: COO incidence matrix of the
    speaker hypergraph; row 0 = node index, row 1 = hyperedge index. All
    utterances of one speaker in the target form one hyperedge.
  - `dialogue_hyperedge_index`  [2, E_d]: every sample's (doc's) utterances
    form one hyperedge, giving high-order *cross-speaker* association inside
    a discussion thread.
  - `temporal_edge_index` / `reply_candidate_edge_index`  [2, E]: standard
    adjacency (optionally bidirectional temporal chain; reply candidates).

The COO `hyperedge_index` is exactly the sparse encoding of the incidence
matrix H consumed by PyG's HypergraphConv.

Each target graph is small (a few hundred nodes), so one full graph per
target is fed at once; the collate function offsets indices per graph so a
batch becomes one disconnected big graph.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.dataset_graph_3 import (
    TargetKnowledgeStore,
    UtteranceEmbeddingStore,
    _load_augmented_records,
    build_target_graph as build_target_graph_v3,
)

logger = logging.getLogger(__name__)


def _ptr_to_hyperedge_coo(
    hyperedge_node_index: torch.Tensor, hyperedge_ptr: torch.Tensor
) -> torch.Tensor:
    """Convert v3's CSR hyperedge storage (node_index + ptr) to COO [2, E].

    Members are stored grouped by hyperedge, so (node, e) pairs follow
    directly from the ptr boundaries.
    """
    if hyperedge_node_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    counts = hyperedge_ptr[1:] - hyperedge_ptr[:-1]
    e_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.long), counts
    )
    return torch.stack([hyperedge_node_index, e_ids], dim=0)


def _dialogue_hyperedge_coo(
    sample_records: List[Dict[str, Any]],
    utterance_ids: torch.Tensor,
) -> torch.Tensor:
    """One hyperedge per sample: all its (global) utterances.

    Members are resolved through the unique-node map (utterance_ids[row] ==
    global id), so a shared sentence occurring in many docs joins many
    dialogue hyperedges -- that is where the cross-speaker high-order
    association lives.
    """
    uid_to_node = {int(u): i for i, u in enumerate(utterance_ids.tolist())}
    # one hyperedge per doc; keep deterministic order
    records = sorted({r["doc_id"]: r for r in sample_records}.values(), key=lambda r: r["doc_id"])
    src_list: List[int] = []
    dst_list: List[int] = []
    for e, rec in enumerate(records):
        for gid in rec["global_ids"]:
            src_list.append(uid_to_node[int(gid)])
            dst_list.append(e)
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)


@dataclass
class TargetGraph:
    """PyG-format target graph (one per target, one node per unique sentence)."""

    target_id: int
    target_text: str = ""

    # Node attributes (all length N_target)
    x: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0)))
    y: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    speaker_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    utterance_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    dialogue_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    turn_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    # CSD target type: "c" = claim target, "n" = noun-phrase target.
    target_type: str = ""

    # Target-side stance semantic grounding prior: [4, D] frozen RoBERTa rows
    # [raw target text; background; favor; against] (zeros((0, 0)) when the
    # knowledge npy is not configured). Row 0 is the un-grounded target
    # representation t_i; rows 1..3 form K = [k^B; k^F; k^A] for grounding.
    target_knowledge: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0)))
    # Related noun-target prior for claim graphs (zeros when unpaired).
    related_noun_knowledge: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0)))
    claim_negated: int = 0

    # Per-doc scoring mapping: CSD scores each doc by its final sentence.
    doc_final_doc_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    doc_final_node_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    doc_final_turn_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    doc_final_true: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))

    # Topologies (PyG standard sparse formats)
    speaker_hyperedge_index: torch.Tensor = field(default_factory=lambda: torch.zeros((2, 0), dtype=torch.long))
    dialogue_hyperedge_index: torch.Tensor = field(default_factory=lambda: torch.zeros((2, 0), dtype=torch.long))
    temporal_edge_index: torch.Tensor = field(default_factory=lambda: torch.zeros((2, 0), dtype=torch.long))
    reply_candidate_edge_index: torch.Tensor = field(default_factory=lambda: torch.zeros((2, 0), dtype=torch.long))

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_text": self.target_text,
            "target_type": self.target_type,
            "target_knowledge": self.target_knowledge,
            "related_noun_knowledge": self.related_noun_knowledge,
            "claim_negated": self.claim_negated,
            "x": self.x,
            "y": self.y,
            "speaker_ids": self.speaker_ids,
            "utterance_ids": self.utterance_ids,
            "dialogue_ids": self.dialogue_ids,
            "turn_index": self.turn_index,
            "doc_final_doc_ids": self.doc_final_doc_ids,
            "doc_final_node_index": self.doc_final_node_index,
            "doc_final_turn_index": self.doc_final_turn_index,
            "doc_final_true": self.doc_final_true,
            "speaker_hyperedge_index": self.speaker_hyperedge_index,
            "dialogue_hyperedge_index": self.dialogue_hyperedge_index,
            "temporal_edge_index": self.temporal_edge_index,
            "reply_candidate_edge_index": self.reply_candidate_edge_index,
            "num_nodes": self.num_nodes,
        }


def build_target_graph(
    target_id: int,
    target_text: str,
    sample_records: List[Dict[str, Any]],
    embedding_store: UtteranceEmbeddingStore,
    bidirectional_temporal: bool = False,
) -> TargetGraph:
    """Build one target graph with one node per unique sentence (PyG formats).

    Reuses the v3 unique-node construction (canonical first-occurrence
    label/speaker; temporal + reply edges unioned over occurrences) and
    exposes the topology as PyG-standard sparse tensors:

    - speaker hyperedges   : v3's CSR speaker groups -> COO
    - dialogue hyperedges  : every doc's utterances form one hyperedge
    - temporal/reply edges : as-is (standard [2, E] adjacency)
    """
    g3 = build_target_graph_v3(
        target_id=target_id,
        target_text=target_text,
        sample_records=sample_records,
        embedding_store=embedding_store,
        bidirectional_temporal=bidirectional_temporal,
    )
    return TargetGraph(
        target_id=target_id,
        target_text=target_text,
        target_type=getattr(g3, "target_type", ""),
        x=g3.x,
        y=g3.y,
        speaker_ids=g3.speaker_ids,
        utterance_ids=g3.utterance_ids,
        dialogue_ids=g3.dialogue_ids,
        turn_index=g3.turn_index,
        doc_final_doc_ids=g3.doc_final_doc_ids,
        doc_final_node_index=g3.doc_final_node_index,
        doc_final_turn_index=g3.doc_final_turn_index,
        doc_final_true=g3.doc_final_true,
        speaker_hyperedge_index=_ptr_to_hyperedge_coo(
            g3.speaker_hyperedge_node_index, g3.speaker_hyperedge_ptr
        ),
        dialogue_hyperedge_index=_dialogue_hyperedge_coo(
            sample_records, g3.utterance_ids
        ),
        temporal_edge_index=g3.temporal_edge_index.clone(),
        reply_candidate_edge_index=g3.reply_candidate_edge_index.clone(),
    )


class TargetGraphDataset(Dataset):
    """Target-level graph dataset exposing PyG-standard sparse topologies."""

    def __init__(
        self,
        raw_dir: str,
        embedding_npy: str,
        max_nodes_per_target: Optional[int] = None,
        bidirectional_temporal: bool = False,
        seed: int = 1234,
        target_knowledge_npy: Optional[str] = None,
    ):
        self.raw_dir = raw_dir
        self.embedding_store = UtteranceEmbeddingStore(embedding_npy)
        self.max_nodes_per_target = max_nodes_per_target
        self.seed = seed

        knowledge_store = None
        if target_knowledge_npy:
            index_path = str(Path(target_knowledge_npy).with_name("target_knowledge_index.json"))
            knowledge_store = TargetKnowledgeStore(target_knowledge_npy, index_path)
            if knowledge_store.dim != self.embedding_store.dim:
                raise ValueError(
                    "target knowledge dim (%d) must equal the utterance embedding dim (%d); "
                    "encode both with the same frozen PLM" % (knowledge_store.dim, self.embedding_store.dim)
                )

        records = _load_augmented_records(raw_dir)
        by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_target[rec["target"]].append(rec)

        self.target_ids: List[int] = []
        self.target_texts: List[str] = []
        self.target_types: List[str] = []
        self.graphs: List[TargetGraph] = []
        n_missing = 0
        for target, recs in sorted(by_target.items()):
            tid = len(self.graphs)
            graph = build_target_graph(
                target_id=tid,
                target_text=target,
                sample_records=recs,
                embedding_store=self.embedding_store,
                bidirectional_temporal=bidirectional_temporal,
            )
            if knowledge_store is not None:
                knowledge = knowledge_store.get(target)
                if knowledge.abs().sum() == 0:
                    n_missing += 1
                graph.target_knowledge = knowledge
            self.target_ids.append(tid)
            self.target_texts.append(target)
            self.target_types.append(graph.target_type)
            self.graphs.append(graph)

        from datasets.claim_noun import attach_claim_noun_fields

        attach_claim_noun_fields(self.graphs, knowledge_store)

        if knowledge_store is not None and n_missing:
            logger.warning(
                "%d/%d targets missing from the knowledge index; using zero priors for them",
                n_missing,
                len(self.graphs),
            )

        logger.info(
            "built %d target graphs from %s (embeddings dim=%d%s)",
            len(self.graphs),
            raw_dir,
            self.embedding_store.dim,
            ", knowledge dim=%d" % knowledge_store.dim if knowledge_store is not None else "",
        )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        graph = self.graphs[idx]
        if self.max_nodes_per_target is not None and graph.num_nodes > self.max_nodes_per_target:
            rng = random.Random(self.seed + idx)
            selected = self._sample_node_centered(graph, rng)
            return self._subgraph_of(graph, selected).to_dict()
        return graph.to_dict()

    # ------------------------------------------------------------------
    # Node-centered subgraph sampling (used only when a target is too big)
    # ------------------------------------------------------------------
    def _sample_node_centered(self, graph: TargetGraph, rng: random.Random) -> List[int]:
        n = graph.num_nodes
        cap = self.max_nodes_per_target or n
        center = rng.randrange(n)
        adj: Dict[int, List[int]] = defaultdict(list)
        for edge_index in (graph.temporal_edge_index, graph.reply_candidate_edge_index):
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            for s, d in zip(src, dst):
                adj[s].append(d)
                adj[d].append(s)
        visited = {center}
        frontier = [center]
        while frontier and len(visited) < cap:
            nxt: List[int] = []
            for u in frontier:
                for v in adj.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
                        if len(visited) >= cap:
                            break
                if len(visited) >= cap:
                    break
            frontier = nxt
        return sorted(visited)

    @staticmethod
    def _subgraph_of(graph: TargetGraph, selected: Sequence[int]) -> TargetGraph:
        selected = sorted(int(i) for i in selected)
        old_to_new = {old: new for new, old in enumerate(selected)}
        sel = torch.tensor(selected, dtype=torch.long)

        def reindex_edges(edge_index: torch.Tensor) -> torch.Tensor:
            if edge_index.size(1) == 0:
                return edge_index.clone()
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            new_src: List[int] = []
            new_dst: List[int] = []
            for s, d in zip(src, dst):
                if s in old_to_new and d in old_to_new:
                    new_src.append(old_to_new[s])
                    new_dst.append(old_to_new[d])
            if not new_src:
                return torch.zeros((2, 0), dtype=torch.long)
            return torch.tensor([new_src, new_dst], dtype=torch.long)

        def remap_hyperedge_coo(coo: torch.Tensor) -> torch.Tensor:
            """Keep only surviving members; renumber hyperedges densely."""
            if coo.size(1) == 0:
                return coo.clone()
            members: Dict[int, List[int]] = defaultdict(list)
            for v, e in zip(coo[0].tolist(), coo[1].tolist()):
                nv = old_to_new.get(v)
                if nv is not None:
                    members[e].append(nv)
            new_src: List[int] = []
            new_dst: List[int] = []
            for new_e, (_, vs) in enumerate(sorted(members.items())):
                for nv in vs:
                    new_src.append(nv)
                    new_dst.append(new_e)
            if not new_src:
                return torch.zeros((2, 0), dtype=torch.long)
            return torch.tensor([new_src, new_dst], dtype=torch.long)

        keep_docs = [
            (doc, old) for doc, old in zip(
                graph.doc_final_doc_ids.tolist(), graph.doc_final_node_index.tolist()
            ) if old in old_to_new
        ]
        keep_turns = [
            t for t, old in zip(graph.doc_final_turn_index.tolist(), graph.doc_final_node_index.tolist())
            if old in old_to_new
        ]
        keep_true = [
            t for t, old in zip(graph.doc_final_true.tolist(), graph.doc_final_node_index.tolist())
            if old in old_to_new
        ]

        return TargetGraph(
            target_id=graph.target_id,
            target_text=graph.target_text,
            target_type=graph.target_type,
            target_knowledge=graph.target_knowledge,
            related_noun_knowledge=graph.related_noun_knowledge,
            claim_negated=graph.claim_negated,
            x=graph.x[sel],
            y=graph.y[sel],
            speaker_ids=graph.speaker_ids[sel],
            utterance_ids=graph.utterance_ids[sel],
            dialogue_ids=graph.dialogue_ids[sel],
            turn_index=graph.turn_index[sel],
            doc_final_doc_ids=torch.tensor([d for d, _ in keep_docs], dtype=torch.long),
            doc_final_node_index=torch.tensor([old_to_new[old] for _, old in keep_docs], dtype=torch.long),
            doc_final_turn_index=torch.tensor(keep_turns, dtype=torch.long),
            doc_final_true=torch.tensor(keep_true, dtype=torch.long),
            speaker_hyperedge_index=remap_hyperedge_coo(graph.speaker_hyperedge_index),
            dialogue_hyperedge_index=remap_hyperedge_coo(graph.dialogue_hyperedge_index),
            temporal_edge_index=reindex_edges(graph.temporal_edge_index),
            reply_candidate_edge_index=reindex_edges(graph.reply_candidate_edge_index),
        )


def target_graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge target graphs into one disconnected batch (manual offsets).

    Ordinary edge indices are offset by the node offset. Hyperedge COO
    indices are offset on **both rows**: row 0 by the node offset, row 1 by
    the running hyperedge count, so COO ids stay valid across the batch for
    PyG's HypergraphConv.
    """
    if not batch:
        return {}

    num_graphs = len(batch)
    node_counts = [item["num_nodes"] for item in batch]
    node_offsets: List[int] = []
    st = 0
    for c in node_counts:
        node_offsets.append(st)
        st += c
    graph_ptr = torch.tensor(node_offsets + [st], dtype=torch.long)

    x_list, y_list = [], []
    speaker_ids_list, utterance_ids_list = [], []
    dialogue_ids_list, turn_index_list = [], []
    doc_final_doc_ids_list, doc_final_node_index_list = [], []
    doc_final_turn_index_list, doc_final_true_list = [], []
    target_ids_list, knowledge_list = [], []
    related_noun_list, type_ids_list, claim_neg_list = [], [], []
    temporal_list, reply_list = [], []
    speaker_he_list, dialogue_he_list = [], []
    speaker_he_off = 0
    dialogue_he_off = 0

    for item, off in zip(batch, node_offsets):
        x_list.append(item["x"])
        y_list.append(item["y"])
        speaker_ids_list.append(item["speaker_ids"])
        utterance_ids_list.append(item["utterance_ids"])
        dialogue_ids_list.append(item["dialogue_ids"])
        turn_index_list.append(item["turn_index"])
        doc_final_doc_ids_list.append(item["doc_final_doc_ids"])
        doc_final_node_index_list.append(item["doc_final_node_index"] + off)
        doc_final_turn_index_list.append(item["doc_final_turn_index"])
        doc_final_true_list.append(item["doc_final_true"])
        target_ids_list.append(torch.full((item["num_nodes"],), item["target_id"], dtype=torch.long))
        knowledge_list.append(item.get("target_knowledge", torch.zeros((0, 0))))
        related_noun_list.append(item.get("related_noun_knowledge", torch.zeros((0, 0))))
        ttype = str(item.get("target_type", "") or "")
        type_id = 1 if ttype == "c" else 0
        type_ids_list.append(torch.full((item["num_nodes"],), type_id, dtype=torch.long))
        claim_neg_list.append(int(item.get("claim_negated", 0)))

        temporal_list.append(item["temporal_edge_index"] + off)
        reply_list.append(item["reply_candidate_edge_index"] + off)

        sh = item["speaker_hyperedge_index"]
        if sh.numel() > 0:
            speaker_he_list.append(sh + torch.tensor([off, speaker_he_off], dtype=torch.long).view(2, 1))
            speaker_he_off += int(sh[1].max().item()) + 1

        dh = item["dialogue_hyperedge_index"]
        if dh.numel() > 0:
            dialogue_he_list.append(dh + torch.tensor([off, dialogue_he_off], dtype=torch.long).view(2, 1))
            dialogue_he_off += int(dh[1].max().item()) + 1

    def cat2(edges: List[torch.Tensor]) -> torch.Tensor:
        if edges:
            return torch.cat(edges, dim=1)
        return torch.zeros((2, 0), dtype=torch.long)

    def cat1(ts: List[torch.Tensor], empty_n: int = 0) -> torch.Tensor:
        if ts:
            return torch.cat(ts, dim=0)
        return torch.zeros((empty_n,), dtype=torch.long)

    return {
        "x": torch.cat(x_list, dim=0),
        "y": torch.cat(y_list, dim=0),
        "speaker_ids": torch.cat(speaker_ids_list, dim=0),
        "utterance_ids": torch.cat(utterance_ids_list, dim=0),
        "dialogue_ids": torch.cat(dialogue_ids_list, dim=0),
        "turn_index": torch.cat(turn_index_list, dim=0),
        "doc_final_doc_ids": cat1(doc_final_doc_ids_list),
        "doc_final_node_index": cat1(doc_final_node_index_list),
        "doc_final_turn_index": cat1(doc_final_turn_index_list),
        "doc_final_true": cat1(doc_final_true_list),
        "target_ids": torch.cat(target_ids_list, dim=0),
        "graph_ptr": graph_ptr,
        "num_graphs": num_graphs,
        "target_knowledge": (
            torch.stack(knowledge_list, dim=0) if knowledge_list else torch.zeros((0, 4, 0))
        ),
        "related_noun_knowledge": (
            torch.stack(related_noun_list, dim=0) if related_noun_list else torch.zeros((0, 4, 0))
        ),
        "target_type_ids": torch.cat(type_ids_list, dim=0) if type_ids_list else torch.zeros((0,), dtype=torch.long),
        "claim_negated": torch.tensor(claim_neg_list, dtype=torch.long),
        "speaker_hyperedge_index": cat2(speaker_he_list),
        "dialogue_hyperedge_index": cat2(dialogue_he_list),
        "temporal_edge_index": cat2(temporal_list),
        "reply_candidate_edge_index": cat2(reply_list),
    }
