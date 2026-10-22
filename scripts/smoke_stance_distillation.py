#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stance KD smoke tests: mock Teacher JSON, attach, L = CE + λ T² KL."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_stance_teacher import StanceTeacher, evaluate_teacher_dir, generate_for_conversation
from modules.stance_distillation import mock_teacher_probs, normalize_stance_probs, teacher_argmax


def test_normalize():
    probs = normalize_stance_probs({"favor": 2, "against": 1, "neutral": 1})
    assert abs(sum(probs) - 1.0) < 1e-6
    assert teacher_argmax(probs) == 0
    print("normalize ok", probs)


def test_mock_generate_and_quality():
    rec = {
        "doc_id": 7,
        "target": "vaccines",
        "target_type": "c",
        "label": 1,
        "sentences": ["hello", "I oppose this"],
        "speakers": [0, 1],
        "global_ids": [10, 11],
        "all_labels": [2, 1],
    }
    teacher = StanceTeacher(provider="mock")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        payload = generate_for_conversation(
            rec, teacher, skip_existing=False, output_dir=out, scope="last_sentence"
        )
        assert payload is not None
        assert payload["utterances"][0]["is_last"] is True
        assert payload["utterances"][0]["global_id"] == 11
        assert payload["utterances"][0]["pred"] == 1
        written = json.loads((out / "conversation_7.json").read_text(encoding="utf-8"))
        assert written["utterances"][0]["global_id"] == 11
        assert written["utterances"][0]["is_last"] is True
        metrics = evaluate_teacher_dir(out)
        assert metrics["mixed_num"] == 1
        assert metrics["mixed_f1_macro"] == 1.0
    print("mock generate+quality ok")


def test_kd_loss_if_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("skip kd loss (no torch)")
        return
    from modules.stance_distillation import stance_kd_loss

    logits = torch.tensor([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1]], dtype=torch.float32)
    teacher = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float32)
    mask = torch.tensor([True, True])
    loss = stance_kd_loss(logits, teacher, mask, temperature=2.0)
    assert torch.isfinite(loss) and float(loss) >= 0.0
    empty = stance_kd_loss(logits, teacher, torch.zeros(2, dtype=torch.bool), temperature=2.0)
    assert float(empty) == 0.0
    ce = F.cross_entropy(logits, torch.tensor([0, 1]))
    total = ce + 0.5 * loss
    assert torch.isfinite(total)
    print("kd loss ok", float(loss), float(total))


def test_mock_probs_peak_on_gold():
    p = mock_teacher_probs(2)
    assert p[2] > p[0] and p[2] > p[1]
    print("mock prior ok", p)


if __name__ == "__main__":
    test_normalize()
    test_mock_generate_and_quality()
    test_kd_loss_if_torch()
    test_mock_probs_peak_on_gold()
    print("smoke_stance_distillation ok")
