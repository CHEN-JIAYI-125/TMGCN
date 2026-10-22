#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrap the original TargetGraphDataset with stance-Teacher tensors.

Does not replace ``dataset_graph_4.py``. Graph topology and labels stay
exactly as the original builder produced them.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from datasets.dataset_graph_4 import target_graph_collate_fn
from datasets.stance_teacher import attach_stance_teacher, load_stance_teacher_store

logger = logging.getLogger(__name__)


class TargetGraphDatasetStanceKD:
    """Proxy dataset: original ``__getitem__`` plus Teacher probs/mask."""

    def __init__(
        self,
        base: Any,
        teacher_path: str,
        scope: str = "last_sentence",
        num_classes: int = 3,
    ):
        self.base = base
        self.teacher_path = teacher_path or ""
        self.scope = scope
        self.num_classes = int(num_classes)
        self.store = load_stance_teacher_store(self.teacher_path)
        covered = 0
        total = 0
        graphs = getattr(self.base, "graphs", None)
        if graphs:
            iterable = ((g.utterance_ids, g.num_nodes) for g in graphs)
        else:
            iterable = (
                (self.base[i]["utterance_ids"], self.base[i]["num_nodes"])
                for i in range(len(self.base))
            )
        for utterance_ids, num_nodes in iterable:
            _, mask = attach_stance_teacher(
                utterance_ids, self.store, scope=self.scope, num_classes=self.num_classes
            )
            covered += int(mask.sum().item())
            total += int(num_nodes)
        pct = 100.0 * covered / max(1, total)
        logger.info(
            "stance teacher node coverage %d/%d (%.1f%%) scope=%s path=%s",
            covered,
            total,
            pct,
            self.scope,
            self.teacher_path,
        )
        if covered == 0:
            logger.warning("stance teacher coverage is 0%%; L_KD will be 0")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        probs, mask = attach_stance_teacher(
            item["utterance_ids"],
            self.store,
            scope=self.scope,
            num_classes=self.num_classes,
        )
        item["stance_teacher_probs"] = probs
        item["stance_teacher_mask"] = mask
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def wrap_with_stance_teacher(
    dataset: Any,
    teacher_path: str,
    scope: str = "last_sentence",
    num_classes: int = 3,
) -> TargetGraphDatasetStanceKD:
    if isinstance(dataset, TargetGraphDatasetStanceKD):
        return dataset
    return TargetGraphDatasetStanceKD(
        dataset, teacher_path=teacher_path, scope=scope, num_classes=num_classes
    )


def stance_kd_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Original collate, then cat Teacher tensors along the node axis."""
    out = target_graph_collate_fn(batch)
    if not batch or "stance_teacher_probs" not in batch[0]:
        return out
    out["stance_teacher_probs"] = torch.cat([item["stance_teacher_probs"] for item in batch], dim=0)
    out["stance_teacher_mask"] = torch.cat([item["stance_teacher_mask"] for item in batch], dim=0)
    return out
