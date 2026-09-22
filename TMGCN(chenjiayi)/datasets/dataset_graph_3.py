#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/6/8

"""Target-level graph dataset for the offline-embedding stance model.

This dataset works on the outputs of `graph/offline/build_embeddings.py`:

- `raw_with_global_ids/*.json`  : original samples augmented with a
  `global_ids` field aligned with `sentences`.
- `embeddings.npy`              : float32 matrix [N, D]; row `i` is the
  embedding of `global_id == i`.

Design notes
------------
- Every utterance in the whole dataset is a node in exactly one target graph.
- Node features are **frozen** embeddings; there is no trainable embedding
  table keyed by utterance id (that would be transductive leakage).
- A graph exists per target, and train/test targets are disjoint. Target ids
  are graph-instance identifiers rather than trainable semantic embeddings:
  graph propagation happens inside a target only, and the model parameters
  are learned from training-target supervision only.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IGNORE_LABEL = -100


class UtteranceEmbeddingStore:
    """Fixed input-feature store for frozen utterance embeddings.

    Loads the offline-computed embedding matrix once and serves rows by
    utterance index. This store is intentionally read-only: it is never used
    to create a trainable `nn.Embedding(utterance_id)`.
    """

    def __init__(self, npy_path: str):
        self.npy_path = str(npy_path)
        arr = np.load(self.npy_path)
        if arr.ndim != 2:
            raise ValueError(f"embeddings must be a [N, D] matrix, got shape {arr.shape}")
        self.data = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).float()
        logger.info(
            "loaded frozen utterance embeddings from %s: %s",
            self.npy_path,
            tuple(self.data.shape),
        )

    @property
    def size(self) -> int:
        return int(self.data.size(0))

    @property
    def dim(self) -> int:
        return int(self.data.size(1))

    def get(self, utterance_ids: torch.Tensor) -> torch.Tensor:
        """Return embeddings for the given utterance indices."""
        ids = utterance_ids
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        ids = ids.view(-1).long()
        if ids.numel() > 0 and (ids.min() < 0 or ids.max() >= self.size):
            raise IndexError(
                f"utterance ids out of range [0, {self.size}): min={ids.min().item()}, max={ids.max().item()}"
            )
        return self.data[ids]


class TargetKnowledgeStore:
    """Frozen per-target stance-view embeddings for semantic grounding.

    Loads `target_knowledge.npy` ([T, 4, D]: raw target; background; favor;
    against) and `target_knowledge_index.json` ({target_text: row}), both
    produced by build_embeddings.py --stage targets with the SAME frozen PLM
    and pooling as the utterance embeddings, so all vectors share one
    representation space. Read-only prior; targets missing from the index
    get zero rows (model can learn to gate them off).
    """

    def __init__(self, npy_path: str, index_path: str):
        arr = np.load(npy_path)
        if arr.ndim != 3 or arr.shape[1] != 4:
            raise ValueError(
                f"target knowledge must be a [T, 4, D] matrix, got shape {tuple(arr.shape)}"
            )
        self.data = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).float()
        with open(index_path, "r", encoding="utf-8") as f:
            self.index: Dict[str, int] = json.load(f)
        logger.info(
            "loaded frozen target knowledge from %s: %s (%d indexed targets)",
            npy_path,
            tuple(self.data.shape),
            len(self.index),
        )

    @property
    def dim(self) -> int:
        return int(self.data.size(2))

    def get(self, target_text: str) -> torch.Tensor:
        """Return the [4, D] knowledge block; zeros if target is unknown."""
        row = self.index.get(target_text)
        if row is None or not (0 <= row < self.data.size(0)):
            return torch.zeros(4, self.dim)
        return self.data[row]


@dataclass
class TargetGraph:
    """One graph per target over all of that target's utterances."""

    target_id: int
    target_text: str = ""
    # CSD target type: "c" = claim target, "n" = noun-phrase target.
    # All docs of one target share it; scoped (c/n/mixed) metrics are
    # computed per type at evaluation time.
    target_type: str = ""

    # Node attributes (all length N_target)
    x: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0)))
    y: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    speaker_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    utterance_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    dialogue_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    turn_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))

    # Per-doc bookkeeping: every sample (doc) is scored by its last sentence.
    # These arrays give, for each doc, the node id of the final sentence and
    # the doc's own label, so evaluation happens on unique nodes but stays
    # aligned with the per-doc CSD metric.
    doc_final_doc_ids: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    doc_final_node_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    doc_final_turn_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    doc_final_true: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))

    # Topologies
    #   speaker hyperedges: nodes are stored sorted by (dialogue_id, turn) so
    #   that a causal (past-only) aggregation can be computed inside the model.
    speaker_hyperedge_node_index: torch.Tensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    speaker_hyperedge_ptr: torch.Tensor = field(default_factory=lambda: torch.zeros((1,), dtype=torch.long))
    temporal_edge_index: torch.Tensor = field(default_factory=lambda: torch.zeros((2, 0), dtype=torch.long))
    reply_candidate_edge_index: torch.Tensor = field(default_factory=lambda: torch.zeros((2, 0), dtype=torch.long))

    # Bookkeeping
    node_dialogue_turn: List[tuple] = field(default_factory=list)  # (dialogue_id, turn_index) per node

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_text": self.target_text,
            "target_type": self.target_type,
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
            "speaker_hyperedge_node_index": self.speaker_hyperedge_node_index,
            "speaker_hyperedge_ptr": self.speaker_hyperedge_ptr,
            "temporal_edge_index": self.temporal_edge_index,
            "reply_candidate_edge_index": self.reply_candidate_edge_index,
            "num_nodes": self.num_nodes,
        }


def _load_augmented_records(raw_path: str) -> List[Dict[str, Any]]:
    """Load raw samples augmented with `global_ids` by the offline stage.

    `raw_path` may be a directory (all *.json inside) or a single JSON file.
    """
    raw_path = Path(raw_path)
    if raw_path.is_dir():
        json_paths = sorted(raw_path.glob("*.json"))
    elif raw_path.is_file() and raw_path.suffix.lower() == ".json":
        json_paths = [raw_path]
    else:
        json_paths = []
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found at {raw_path}")
    records: List[Dict[str, Any]] = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            sentences = item.get("sentences", []) or []
            if not sentences:
                continue
            global_ids = item.get("global_ids")
            if global_ids is None or len(global_ids) != len(sentences):
                raise ValueError(
                    f"record id={item.get('id')} in {path.name} has no aligned global_ids "
                    f"(len={len(global_ids) if global_ids is not None else 'missing'}, sentences={len(sentences)}); "
                    "run graph/offline/build_embeddings.py --stage dedup first"
                )
            records.append(
                {
                    "doc_id": int(item.get("id", item.get("doc_id", -1))),
                    "target": item.get("target", ""),
                    "target_type": item.get("target_type", ""),
                    "label": int(item.get("label", -1)),
                    "sentences": list(sentences),
                    "global_ids": [int(g) for g in global_ids],
                    "speakers": [int(s) for s in (item.get("speakers", []) or [])],
                    "all_labels": [int(s) for s in (item.get("all_labels", []) or [])],
                    "source": path.stem,
                }
            )
    logger.info("loaded %d augmented samples from %s", len(records), raw_path)
    return records


def build_target_graph(
    target_id: int,
    target_text: str,
    sample_records: List[Dict[str, Any]],
    embedding_store: UtteranceEmbeddingStore,
    bidirectional_temporal: bool = False,
) -> TargetGraph:
    """Build one target graph with **one node per unique sentence**.

    A sentence (identified by its global utterance id) may appear in many
    sample chains of the target: it is then a *single node* whose neighbours
    are collected from every occurrence, not one node per occurrence.

    Node resolution
    ---------------
    - One node per unique `utterance_id`, in order of first occurrence
      (sample records are processed by ascending doc_id, turns in order).
    - Node feature  = the frozen embedding of that sentence.
    - Node label/speaker = the value of the *first occurrence* (per design
      decision; inconsistent later occurrences are counted and logged).
    - Node `dialogue_id`/`turn_index` = first occurrence, only bookkeeping.

    Edges
    -----
    - Temporal: for every sample chain, each consecutive pair of unique
      sentences becomes one edge (deduplicated).
    - Reply candidates: for every sample chain, every antecedent pair
      (j < t) becomes a candidate edge (deduplicated). A shared sentence
      therefore accumulates the union of its antecedents over all samples
      where it appears - the target-level global context.

    Per-doc evaluation mapping
    --------------------------
    CSD scores each sample by its last sentence. Because nodes are shared,
    a doc's final sentence may be a node that first occurred elsewhere, so
    the doc -> final node mapping is kept explicitly in `doc_final_*`.
    """
    records = sorted(sample_records, key=lambda r: (r["doc_id"],))

    # target_type is a property of the target; warn if the group disagrees
    type_counter = Counter(str(r.get("target_type", "")) for r in records)
    if len(type_counter) > 1:
        logger.warning(
            "target=%r: mixed target_type %s; scoped metrics will use the majority %r",
            target_text,
            dict(type_counter),
            type_counter.most_common(1)[0][0],
        )

    # uid -> canonical (first occurrence) attributes, insertion order = node order
    uid_to_node: Dict[int, int] = {}
    node_uids: List[int] = []
    node_speakers: List[int] = []
    node_labels: List[int] = []
    node_first_doc: List[int] = []
    node_first_turn: List[int] = []

    doc_final_doc_ids: List[int] = []
    doc_final_node_index: List[int] = []
    doc_final_turn_index: List[int] = []
    doc_final_true: List[int] = []
    label_conflicts = 0

    for rec in records:
        doc_id = rec["doc_id"]
        gids = rec["global_ids"]
        speakers = rec["speakers"]
        all_labels = rec["all_labels"]
        doc_label = rec["label"]

        for turn, gid in enumerate(gids):
            speaker = speakers[turn] if turn < len(speakers) else -1
            occ_label = all_labels[turn] if turn < len(all_labels) else doc_label
            if gid not in uid_to_node:
                uid_to_node[gid] = len(node_uids)
                node_uids.append(gid)
                node_speakers.append(speaker)
                node_labels.append(occ_label)
                node_first_doc.append(doc_id)
                node_first_turn.append(turn)
            else:
                # first-occurrence policy: canonical label/speaker are kept;
                # later disagreements are only diagnosed.
                idx = uid_to_node[gid]
                if (
                    node_labels[idx] >= 0
                    and occ_label >= 0
                    and node_labels[idx] != occ_label
                ):
                    label_conflicts += 1

        # last sentence of this doc = its evaluation anchor
        last_gid = gids[-1]
        last_turn = len(gids) - 1
        doc_final_doc_ids.append(doc_id)
        doc_final_node_index.append(uid_to_node[last_gid])
        doc_final_turn_index.append(last_turn)
        last_occ_label = (
            all_labels[last_turn] if last_turn < len(all_labels) else doc_label
        )
        doc_final_true.append(doc_label if doc_label >= 0 else last_occ_label)

    n = len(node_uids)
    if n == 0:
        return TargetGraph(target_id=target_id, target_text=target_text)
    if label_conflicts > 0:
        logger.warning(
            "target=%r: %d later occurrences disagree with the first-occurrence label",
            target_text,
            label_conflicts,
        )

    # --- node attributes ---
    utterance_ids = torch.tensor(node_uids, dtype=torch.long)
    x = embedding_store.get(utterance_ids)
    y = torch.tensor(
        [lab if lab >= 0 else IGNORE_LABEL for lab in node_labels], dtype=torch.long
    )
    speaker_ids = torch.tensor(node_speakers, dtype=torch.long)
    dialogue_ids = torch.tensor(node_first_doc, dtype=torch.long)
    turn_index = torch.tensor(node_first_turn, dtype=torch.long)

    # --- speaker hyperedges: unique nodes grouped by speaker, in node order ---
    speaker_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for local_idx, spk in enumerate(node_speakers):
        if spk >= 0:
            speaker_to_nodes[spk].append(local_idx)
    hyperedge_node_index: List[int] = []
    hyperedge_ptr = [0]
    for speaker in sorted(speaker_to_nodes.keys()):
        members = sorted(speaker_to_nodes[speaker])
        hyperedge_node_index.extend(members)
        hyperedge_ptr.append(len(hyperedge_node_index))

    # --- edges derived from every occurrence chain, pointing at unique nodes ---
    temporal_pairs: set = set()
    reply_pairs: set = set()
    for rec in records:
        gids = rec["global_ids"]
        chain = [uid_to_node[g] for g in gids]
        for i in range(len(chain) - 1):
            if chain[i] != chain[i + 1]:
                temporal_pairs.add((chain[i], chain[i + 1]))
                if bidirectional_temporal:
                    temporal_pairs.add((chain[i + 1], chain[i]))
        for t in range(len(chain)):
            for j in range(t):
                if chain[j] != chain[t]:
                    reply_pairs.add((chain[j], chain[t]))

    def _pairs_to_edges(pairs) -> torch.Tensor:
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        src = [p[0] for p in sorted(pairs)]
        dst = [p[1] for p in sorted(pairs)]
        return torch.tensor([src, dst], dtype=torch.long)

    return TargetGraph(
        target_id=target_id,
        target_text=target_text,
        target_type=type_counter.most_common(1)[0][0] if type_counter else "",
        x=x,
        y=y,
        speaker_ids=speaker_ids,
        utterance_ids=utterance_ids,
        dialogue_ids=dialogue_ids,
        turn_index=turn_index,
        doc_final_doc_ids=torch.tensor(doc_final_doc_ids, dtype=torch.long),
        doc_final_node_index=torch.tensor(doc_final_node_index, dtype=torch.long),
        doc_final_turn_index=torch.tensor(doc_final_turn_index, dtype=torch.long),
        doc_final_true=torch.tensor(doc_final_true, dtype=torch.long),
        speaker_hyperedge_node_index=torch.tensor(hyperedge_node_index, dtype=torch.long),
        speaker_hyperedge_ptr=torch.tensor(hyperedge_ptr, dtype=torch.long),
        temporal_edge_index=_pairs_to_edges(temporal_pairs),
        reply_candidate_edge_index=_pairs_to_edges(reply_pairs),
        node_dialogue_turn=list(zip(node_first_doc, node_first_turn)),
    )


class TargetGraphDataset(Dataset):
    """Dataset of target-level graphs.

    A sample is one target graph (v1: the full graph; optional
    node-centered subgraph sampling via `max_nodes_per_target`).
    """

    def __init__(
        self,
        raw_dir: str,
        embedding_npy: str,
        max_nodes_per_target: int = 512,
        bidirectional_temporal: bool = False,
        seed: int = 1234,
    ):
        self.raw_dir = raw_dir
        self.embedding_store = UtteranceEmbeddingStore(embedding_npy)
        self.max_nodes_per_target = max_nodes_per_target
        self.bidirectional_temporal = bidirectional_temporal
        self.seed = seed

        records = _load_augmented_records(raw_dir)
        by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_target[rec["target"]].append(rec)

        self.target_ids: List[int] = []
        self.target_texts: List[str] = []
        self.target_types: List[str] = []
        self.graphs: List[TargetGraph] = []
        for target, recs in by_target.items():
            tid = len(self.graphs)
            graph = build_target_graph(
                target_id=tid,
                target_text=target,
                sample_records=recs,
                embedding_store=self.embedding_store,
                bidirectional_temporal=bidirectional_temporal,
            )
            self.target_ids.append(tid)
            self.target_texts.append(target)
            self.target_types.append(graph.target_type)
            self.graphs.append(graph)

        logger.info(
            "built %d target graphs from %s (embeddings dim=%d)",
            len(self.graphs),
            raw_dir,
            self.embedding_store.dim,
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
        # BFS expansion over temporal + reply candidate edges, capped at `cap`
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

        temporal = graph.temporal_edge_index
        reply = graph.reply_candidate_edge_index

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

        hnode = graph.speaker_hyperedge_node_index.tolist()
        hptr = graph.speaker_hyperedge_ptr.tolist()
        new_hnode: List[int] = []
        new_hptr = [0]
        for e in range(len(hptr) - 1):
            members = [old_to_new[m] for m in hnode[hptr[e]:hptr[e + 1]] if m in old_to_new]
            if members:
                new_hnode.extend(members)
                new_hptr.append(len(new_hnode))

        # doc -> final node mapping: drop docs whose final node fell outside
        # the subgraph, remap the rest
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
            speaker_hyperedge_node_index=torch.tensor(new_hnode, dtype=torch.long),
            speaker_hyperedge_ptr=torch.tensor(new_hptr, dtype=torch.long),
            temporal_edge_index=reindex_edges(temporal),
            reply_candidate_edge_index=reindex_edges(reply),
            node_dialogue_turn=[graph.node_dialogue_turn[i] for i in selected],
        )


def target_graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple target-level subgraphs into one disconnected batched graph.

    Produces `graph_ptr` with the node range of every graph in the batch.
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
    target_ids_list = []
    temporal_list, reply_list = [], []
    hnode_list, hptr_list = [], []
    hptr_offset = 0

    for item, offset in zip(batch, node_offsets):
        x_list.append(item["x"])
        y_list.append(item["y"])
        speaker_ids_list.append(item["speaker_ids"])
        utterance_ids_list.append(item["utterance_ids"])
        dialogue_ids_list.append(item["dialogue_ids"])
        turn_index_list.append(item["turn_index"])
        doc_final_doc_ids_list.append(item["doc_final_doc_ids"])
        doc_final_node_index_list.append(item["doc_final_node_index"] + offset)
        doc_final_turn_index_list.append(item["doc_final_turn_index"])
        doc_final_true_list.append(item["doc_final_true"])
        target_ids_list.append(torch.full((item["num_nodes"],), item["target_id"], dtype=torch.long))

        temporal_list.append(item["temporal_edge_index"] + offset)
        reply_list.append(item["reply_candidate_edge_index"] + offset)

        hnode_list.append(item["speaker_hyperedge_node_index"] + offset)
        ptr = item["speaker_hyperedge_ptr"] + hptr_offset
        hptr_list.append(ptr)
        hptr_offset += int(item["speaker_hyperedge_node_index"].numel())

    def cat2(edges: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(edges, dim=1) if edges and edges[0].size(1) > 0 else torch.zeros((2, 0), dtype=torch.long)

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
        "temporal_edge_index": cat2(temporal_list),
        "reply_candidate_edge_index": cat2(reply_list),
        "speaker_hyperedge_node_index": torch.cat(hnode_list, dim=0) if hnode_list else torch.zeros((0,), dtype=torch.long),
        "speaker_hyperedge_ptr": torch.cat(hptr_list, dim=0) if hptr_list else torch.zeros((1,), dtype=torch.long),
    }