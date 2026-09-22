#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""带 Reply Teacher 分布的目标图数据集（dataset_graph_4 的非破坏扩展）。

原 ``dataset_graph_4.TargetGraphDataset`` 构造：
    speaker / dialogue hypergraph + temporal 边 + 全历史等权 reply-candidate 边。

本文件只多做一件事：若 ``teacher_path`` 下存在 LLM Teacher JSON，则为每个
目标图附加

    reply_teacher_distribution  [N, N+1]
    reply_teacher_mask          [N, N+1]  bool

与 Student 输出对齐：第 0 列 = ROOT，第 j+1 列 = 回复图节点 j。
Teacher 是 conversation 级、turn 级的 soft 分布，这里按「节点首次出现的
那条对话」投影到目标图（与 v3 的 first-occurrence 节点策略一致）。

若 teacher 目录不存在、或某条对话没有对应文件：该节点 mask 全 False，
训练时 KD 自动跳过，模型仍可运行（Student 图 / 原 candidate 图照常）。

当 ``reply_distillation=False`` 时请不要使用本数据集，直接走原
``dataset_graph_4``，以保证 baseline 数据管线逐张量等价。
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from datasets.dataset_graph_3 import _load_augmented_records
from datasets.dataset_graph_4 import (
    TargetGraph,
    TargetGraphDataset,
    target_graph_collate_fn as _base_collate_fn,
)

logger = logging.getLogger(__name__)


def load_reply_teacher_store(teacher_path: str) -> Dict[Tuple[int, int], Dict[str, float]]:
    """扫描 ``data/reply_teacher/*.json``，索引为 (conversation_id, utterance_id)。

    单个文件既可以是一条对象，也可以是该 conversation 全部 turn 的列表。
    键名兼容 conversation_id / doc_id / id。
    """
    root = Path(teacher_path)
    store: Dict[Tuple[int, int], Dict[str, float]] = {}
    if not teacher_path or not root.exists():
        logger.warning(
            "reply teacher 路径不存在 (%s)；将不加载 P_T，模型仍按无 Teacher 运行",
            teacher_path,
        )
        return store
    files = sorted(root.glob("*.json"))
    if not files:
        logger.warning("reply teacher 目录 %s 为空", teacher_path)
        return store
    n_bad = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            n_bad += 1
            logger.warning("无法解析 teacher 文件 %s: %s", fp, exc)
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                cid = int(item.get("conversation_id", item.get("doc_id", item.get("id", -1))))
                uid = int(item.get("utterance_id", item.get("turn", item.get("t", -1))))
            except (TypeError, ValueError):
                continue
            dist = item.get("reply_distribution", item.get("distribution", {}))
            if cid < 0 or uid < 0 or not isinstance(dist, dict):
                continue
            store[(cid, uid)] = {str(k): float(v) for k, v in dist.items()}
    logger.info("加载 reply teacher %d 条 (conversation, turn)，坏文件=%d", len(store), n_bad)
    return store


def _normalize_teacher_dist(raw: Dict[str, float], t: int) -> Dict[str, float]:
    """只保留 ROOT + 历史 turn 0..t-1，丢弃未来键，非负归一化。"""
    keys = ["ROOT"] + [str(j) for j in range(t)]
    scores = []
    for k in keys:
        try:
            scores.append(max(0.0, float(raw.get(k, 0.0))))
        except (TypeError, ValueError):
            scores.append(0.0)
    total = sum(scores)
    if total <= 0.0:
        return {}
    return {k: s / total for k, s in zip(keys, scores)}


def attach_reply_teacher(
    graph: TargetGraph,
    sample_records: List[Dict[str, Any]],
    teacher_store: Dict[Tuple[int, int], Dict[str, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 conversation 级 Teacher 分布投影到目标图节点。

    节点采用 first-occurrence：一个全局 utterance 若出现在多段对话中，
    只用它第一次进入该目标图时所属对话的 Teacher。这与节点 label/speaker
    的 first-occurrence 策略对齐，避免同一节点被多条互相矛盾的 P_T 覆盖。

    返回
    ----
    dist : FloatTensor [N, N+1]
    mask : BoolTensor  [N, N+1]  True 表示该位置属于 Teacher support
    """
    n = int(graph.num_nodes)
    dist = torch.zeros((n, n + 1), dtype=torch.float32)
    mask = torch.zeros((n, n + 1), dtype=torch.bool)
    if n == 0 or not teacher_store:
        return dist, mask

    uid_to_node = {int(u): i for i, u in enumerate(graph.utterance_ids.tolist())}
    filled = set()  # 已经写过 Teacher 的图节点
    n_hit = 0
    records = sorted(sample_records, key=lambda r: (r["doc_id"],))
    for rec in records:
        cid = int(rec["doc_id"])
        gids = rec["global_ids"]
        try:
            chain = [uid_to_node[int(g)] for g in gids]
        except KeyError:
            continue
        for t, node_t in enumerate(chain):
            if node_t in filled:
                continue
            raw = teacher_store.get((cid, t))
            if not raw:
                continue
            norm = _normalize_teacher_dist(raw, t)
            if not norm:
                continue
            mask[node_t, 0] = True
            dist[node_t, 0] = float(norm.get("ROOT", 0.0))
            for j in range(t):
                node_j = chain[j]
                mask[node_t, node_j + 1] = True
                dist[node_t, node_j + 1] += float(norm.get(str(j), 0.0))
            # 若 chain 里出现重复节点导致同一列被加了两次，再归一化
            support = dist[node_t] * mask[node_t].to(dist.dtype)
            z = float(support.sum().item())
            if z > 0:
                dist[node_t] = support / z
            filled.add(node_t)
            n_hit += 1
    logger.debug(
        "target=%r teacher 覆盖 %d/%d 节点",
        graph.target_text,
        n_hit,
        n,
    )
    return dist, mask


def _remap_teacher_to_subgraph(
    dist: torch.Tensor,
    mask: torch.Tensor,
    selected: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """节点中心子图采样时，把 [N, N+1] Teacher 矩阵收缩到子图节点。"""
    selected = [int(i) for i in selected]
    n_sub = len(selected)
    new_dist = torch.zeros((n_sub, n_sub + 1), dtype=dist.dtype)
    new_mask = torch.zeros((n_sub, n_sub + 1), dtype=torch.bool)
    if n_sub == 0:
        return new_dist, new_mask
    sel = torch.tensor(selected, dtype=torch.long)
    new_dist[:, 0] = dist[sel, 0]
    new_mask[:, 0] = mask[sel, 0]
    # dist[t, j+1] -> new[new_t, new_j+1]
    hist = dist[sel][:, sel + 1]
    hist_m = mask[sel][:, sel + 1]
    new_dist[:, 1:] = hist
    new_mask[:, 1:] = hist_m
    # 丢掉已经不在子图里的 support 后需要重新归一化
    for i in range(n_sub):
        if not bool(new_mask[i].any()):
            continue
        row = new_dist[i] * new_mask[i].to(new_dist.dtype)
        z = float(row.sum().item())
        if z > 0:
            new_dist[i] = row / z
        else:
            new_mask[i] = False
    return new_dist, new_mask


class TargetGraphDatasetRD(TargetGraphDataset):
    """原目标图数据集 + 每张图的 ``reply_teacher_distribution``。"""

    def __init__(
        self,
        raw_dir: str,
        embedding_npy: str,
        max_nodes_per_target: Optional[int] = None,
        bidirectional_temporal: bool = False,
        seed: int = 1234,
        target_knowledge_npy: Optional[str] = None,
        teacher_path: Optional[str] = None,
        reply_distillation: bool = False,
    ):
        super().__init__(
            raw_dir=raw_dir,
            embedding_npy=embedding_npy,
            max_nodes_per_target=max_nodes_per_target,
            bidirectional_temporal=bidirectional_temporal,
            seed=seed,
            target_knowledge_npy=target_knowledge_npy,
        )
        self.reply_distillation = bool(reply_distillation)
        self.teacher_path = teacher_path or ""
        self.reply_teacher_dist: List[torch.Tensor] = []
        self.reply_teacher_mask: List[torch.Tensor] = []

        if not self.reply_distillation:
            # 关闭蒸馏时不读 teacher，也不往 sample 里塞大矩阵
            return

        store = load_reply_teacher_store(self.teacher_path)
        records = _load_augmented_records(raw_dir)
        by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_target[rec["target"]].append(rec)

        n_cov = 0
        n_nodes = 0
        for graph in self.graphs:
            recs = by_target.get(graph.target_text, [])
            dist, mask = attach_reply_teacher(graph, recs, store)
            self.reply_teacher_dist.append(dist)
            self.reply_teacher_mask.append(mask)
            n_cov += int(mask.any(dim=-1).sum().item())
            n_nodes += int(graph.num_nodes)
        logger.info(
            "reply teacher 节点覆盖 %d/%d (%.1f%%)  path=%s",
            n_cov,
            n_nodes,
            100.0 * n_cov / max(1, n_nodes),
            self.teacher_path,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        if not self.reply_distillation:
            return item
        graph = self.graphs[idx]
        dist = self.reply_teacher_dist[idx]
        mask = self.reply_teacher_mask[idx]
        n_now = int(item["num_nodes"])
        if n_now != graph.num_nodes:
            # 发生了节点中心子图采样：按 utterance_id 对齐选中的节点
            full_uid = graph.utterance_ids.tolist()
            uid_to_old = {int(u): i for i, u in enumerate(full_uid)}
            selected = [uid_to_old[int(u)] for u in item["utterance_ids"].tolist()]
            dist, mask = _remap_teacher_to_subgraph(dist, mask, selected)
        item["reply_teacher_distribution"] = dist
        item["reply_teacher_mask"] = mask
        return item


def target_graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """在原 collate 之上，把每张图的 [n, n+1] Teacher 矩阵拼成 batch 块对角。

    节点 offset 之后：
        teacher_batch[off+t, 0]          = ROOT
        teacher_batch[off+t, off+j+1]    = P_T(j|t)  （图内历史节点）
    跨图位置全 0 / False，Student 的候选边本身也不跨图。
    """
    out = _base_collate_fn(batch)
    if not batch or "reply_teacher_distribution" not in batch[0]:
        return out

    node_counts = [int(item["num_nodes"]) for item in batch]
    n_total = int(sum(node_counts))
    teacher = torch.zeros((n_total, n_total + 1), dtype=torch.float32)
    mask = torch.zeros((n_total, n_total + 1), dtype=torch.bool)
    off = 0
    for item, n in zip(batch, node_counts):
        d = item["reply_teacher_distribution"]
        m = item["reply_teacher_mask"]
        if d.numel() == 0:
            off += n
            continue
        teacher[off : off + n, 0] = d[:, 0]
        mask[off : off + n, 0] = m[:, 0]
        teacher[off : off + n, off + 1 : off + n + 1] = d[:, 1:]
        mask[off : off + n, off + 1 : off + n + 1] = m[:, 1:]
        off += n
    out["reply_teacher_distribution"] = teacher
    out["reply_teacher_mask"] = mask
    return out
