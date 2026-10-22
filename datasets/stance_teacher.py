#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load offline stance-Teacher JSON and attach it to original target graphs.

Teacher files are written by ``scripts/generate_stance_teacher.py``.
Keys are global utterance ids so they line up with ``utterance_ids`` on
the original TMGCN graph. Missing files or missing nodes leave a False
mask; the Student then falls back to CE only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.stance_distillation import normalize_stance_probs

logger = logging.getLogger(__name__)


def _as_prob_list(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return [float(v) for v in normalize_stance_probs(raw)]
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        try:
            vals = [max(0.0, float(raw[0])), max(0.0, float(raw[1])), max(0.0, float(raw[2]))]
        except (TypeError, ValueError):
            return None
        total = sum(vals)
        if total <= 0.0:
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        return [v / total for v in vals]
    return None


def load_stance_teacher_store(teacher_path: str) -> Dict[int, Dict[str, Any]]:
    """Scan ``conversation_*.json`` and index rows by ``global_id``.

    If the same global_id appears more than once, last-sentence rows win;
    otherwise later files overwrite earlier ones after averaging is skipped
    to keep a single explicit Teacher vector per node.
    """
    store: Dict[int, Dict[str, Any]] = {}
    root = Path(teacher_path) if teacher_path else Path()
    if not teacher_path or not root.exists():
        logger.warning("stance teacher path missing (%s); KD will be 0", teacher_path)
        return store

    files = sorted(root.glob("conversation_*.json"))
    if root.is_file() and root.suffix.lower() == ".json":
        files = [root]
    if not files:
        logger.warning("stance teacher directory %s is empty", teacher_path)
        return store

    n_bad = 0
    n_rows = 0
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            n_bad += 1
            logger.warning("cannot parse stance teacher %s: %s", fp, exc)
            continue
        rows = payload.get("utterances", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            n_bad += 1
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            gid = row.get("global_id", row.get("utterance_id"))
            try:
                gid = int(gid)
            except (TypeError, ValueError):
                continue
            probs = _as_prob_list(row.get("probs", row.get("reply_distribution")))
            if probs is None:
                continue
            is_last = bool(row.get("is_last", False))
            prev = store.get(gid)
            if prev is not None and prev.get("is_last") and not is_last:
                continue
            store[gid] = {
                "probs": probs,
                "is_last": is_last,
                "gold": int(row.get("gold_label", row.get("gold", -1))),
                "doc_id": int(row.get("conversation_id", row.get("doc_id", -1))),
                "turn_index": int(row.get("utterance_id", row.get("turn_index", -1))),
            }
            n_rows += 1

    logger.info(
        "loaded stance teacher %d unique global_ids (%d rows, bad_files=%d) path=%s",
        len(store),
        n_rows,
        n_bad,
        teacher_path,
    )
    return store


def attach_stance_teacher(
    utterance_ids: Any,
    store: Dict[int, Dict[str, Any]],
    scope: str = "last_sentence",
    num_classes: int = 3,
) -> Tuple[Any, Any]:
    """Return ``(probs [N, C], mask [N])`` aligned with graph nodes."""
    import torch

    n = int(utterance_ids.numel())
    probs = torch.zeros((n, num_classes), dtype=torch.float32)
    mask = torch.zeros((n,), dtype=torch.bool)
    only_last = str(scope or "last_sentence").strip().lower() not in ("all", "all_turns", "prefix")
    for i, gid in enumerate(utterance_ids.tolist()):
        row = store.get(int(gid))
        if row is None:
            continue
        if only_last and not row.get("is_last", False):
            continue
        probs[i] = torch.tensor(row["probs"], dtype=torch.float32)
        mask[i] = True
    return probs, mask
