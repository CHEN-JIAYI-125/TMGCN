#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""不依赖数据与 GPU 的蒸馏模块冒烟测试。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_reply_teacher import (  # noqa: E402
    LLMReplyTeacher,
    generate_for_conversation,
    normalize_reply_distribution,
)
from modules.reply_distillation import (  # noqa: E402
    ReplyKDLoss,
    ReplyPredictor,
    build_soft_reply_graph,
)


def test_predictor_and_graph():
    torch.manual_seed(0)
    n, d = 4, 16
    h = torch.randn(n, d)
    # 对话因果候选：0->1, 0->2, 1->2, 0->3, 1->3, 2->3 以及一条「节点下标逆序」边 3->1
    # 若误用下三角掩码，3->1 会被丢掉。
    src = torch.tensor([0, 0, 1, 0, 1, 2, 3])
    dst = torch.tensor([1, 2, 2, 3, 3, 3, 1])
    edge_index = torch.stack([src, dst], dim=0)

    pred = ReplyPredictor(hidden_size=d, dropout=0.0)
    pred.eval()
    with torch.no_grad():
        probs, logits, valid = pred(h, candidate_edge_index=edge_index)

    assert probs.shape == (n, n + 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(n), atol=1e-5)
    # ROOT 列合法
    assert torch.isfinite(logits[:, 0]).all()
    # 逆序边 3->1 必须有限（节点 1 回复节点 3）
    assert torch.isfinite(logits[1, 3 + 1]), "conversation-causal edge 3->1 was masked"
    # 自环非法
    for i in range(n):
        assert not torch.isfinite(logits[i, i + 1])
    # 从未出现的候选：2 回复 1 不在边上，必须被掩码
    assert not torch.isfinite(logits[2, 1 + 1]), "non-candidate 1<-2 should be masked"

    edge_i, edge_w = build_soft_reply_graph(probs, threshold=0.0, valid_hist=valid)
    assert edge_i.size(0) == 2
    assert edge_i.size(1) == edge_w.numel()
    # 方向必须是 j -> t
    for j, t, w in zip(edge_i[0].tolist(), edge_i[1].tolist(), edge_w.tolist()):
        assert w == float(probs[t, j + 1])
        assert valid[t, j]
    print("predictor+graph ok", probs.round(decimals=3).tolist())


def test_kd_loss():
    n = 3
    logits = torch.tensor(
        [
            [0.0, float("-inf"), float("-inf"), float("-inf")],
            [0.2, 1.5, float("-inf"), float("-inf")],
            [0.1, 0.3, 2.0, float("-inf")],
        ]
    )
    teacher = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.05, 0.15, 0.80, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    crit = ReplyKDLoss(temperature=3.0)
    loss = crit(logits, teacher, mask)
    assert torch.isfinite(loss) and float(loss) >= 0.0
    # 无 teacher 时为 0
    loss0 = crit(logits, teacher, torch.zeros_like(mask))
    assert float(loss0) == 0.0
    print("kd loss ok", float(loss))


def test_teacher_script_mock():
    dist = normalize_reply_distribution({"ROOT": 1.0, "0": 2.0, "9": 99.0}, t=2)
    assert set(dist) == {"ROOT", "0", "1"}
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert "9" not in dist

    rec = {
        "conversation_id": 42,
        "target": "climate change",
        "utterances": [
            {"speaker": "speaker_0", "text": "hello"},
            {"speaker": "speaker_1", "text": "I disagree"},
            {"speaker": "speaker_0", "text": "why?"},
        ],
    }
    teacher = LLMReplyTeacher(provider="mock")
    with tempfile.TemporaryDirectory() as td:
        rows = generate_for_conversation(rec, teacher, skip_existing=False, output_dir=Path(td))
        assert rows is not None and len(rows) == 3
        assert rows[0]["reply_distribution"] == {"ROOT": 1.0}
        assert "ROOT" in rows[2]["reply_distribution"] and "0" in rows[2]["reply_distribution"]
        payload = json.loads((Path(td) / "conversation_42.json").read_text(encoding="utf-8"))
        assert payload[1]["utterance_id"] == 1
    print("teacher mock ok")


if __name__ == "__main__":
    test_predictor_and_graph()
    test_kd_loss()
    test_teacher_script_mock()
    print("all smoke tests passed")
