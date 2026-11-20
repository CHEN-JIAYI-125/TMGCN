#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Oversample rare (target, class) train docs; originals stay."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.oversample import oversample_records, oversample_summary


def _rec(doc_id, target, label):
    return {
        "id": doc_id,
        "doc_id": doc_id,
        "target": target,
        "label": label,
        "global_ids": [doc_id],
        "sentences": ["x"],
        "speakers": [0],
        "all_labels": [label],
    }


def test_trump_favor_reaches_majority():
    recs = []
    recs += [_rec(i, "Trump", 1) for i in range(20)]  # against
    recs += [_rec(100 + i, "Trump", 2) for i in range(20)]  # none
    recs += [_rec(200 + i, "Trump", 0) for i in range(2)]  # favor
    recs += [_rec(300 + i, "Biden", 0) for i in range(10)]
    out = oversample_records(recs, seed=1234, max_factor=10.0)
    after = oversample_summary(out)["targets"]
    assert after["Trump"]["favor"] == after["Trump"]["none"] == 20
    assert after["Trump"]["against"] == 20
    assert after["Biden"]["favor"] == 10
    orig_ids = {(r["doc_id"], r["target"]) for r in recs}
    out_ids = {(r["doc_id"], r["target"]) for r in out}
    assert orig_ids.issubset(out_ids)
    trump_ids = [r["doc_id"] for r in out if r["target"] == "Trump"]
    assert len(trump_ids) == len(set(trump_ids))
    assert len(out) > len(recs)


def test_cap_max_factor():
    recs = [_rec(i, "Trump", 2) for i in range(20)] + [_rec(100, "Trump", 0)]
    out = oversample_records(recs, seed=0, max_factor=3.0)
    after = oversample_summary(out)["targets"]["Trump"]
    assert after["favor"] == 3
    assert after["none"] == 20


def test_ckpt_mean_favg():
    from main_graph import checkpoint_score

    metrics = {
        "mixed_f1_macro": 0.5,
        "mixed_f1_favg": 0.4,
        "per_target_names": ["Biden", "Trump"],
        "t_Biden_f1_favg": 0.2,
        "t_Trump_f1_favg": 0.4,
    }
    assert abs(checkpoint_score(metrics, "target_favg_mean") - 0.3) < 1e-9
    assert checkpoint_score(metrics, "mixed_f1_macro") == 0.5


if __name__ == "__main__":
    test_trump_favor_reaches_majority()
    test_cap_max_factor()
    test_ckpt_mean_favg()
    print("oversample tests passed")
