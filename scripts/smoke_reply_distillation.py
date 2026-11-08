#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Student 软回复图冒烟测试，不依赖数据与 GPU。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.reply_distillation import (  # noqa: E402
    ReplyPredictor,
    build_soft_reply_graph,
)


def test_predictor_and_graph():
    torch.manual_seed(0)
    n, d = 4, 16
    h = torch.randn(n, d)
    src = torch.tensor([0, 0, 1, 0, 1, 2, 3])
    dst = torch.tensor([1, 2, 2, 3, 3, 3, 1])
    edge_index = torch.stack([src, dst], dim=0)

    pred = ReplyPredictor(hidden_size=d, dropout=0.0)
    pred.eval()
    with torch.no_grad():
        probs, logits, valid = pred(h, candidate_edge_index=edge_index)

    assert probs.shape == (n, n + 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(n), atol=1e-5)
    assert torch.isfinite(logits[:, 0]).all()
    assert torch.isfinite(logits[1, 3 + 1]), "conversation-causal edge 3->1 was masked"
    for i in range(n):
        assert not torch.isfinite(logits[i, i + 1])
    assert not torch.isfinite(logits[2, 1 + 1]), "non-candidate 1<-2 should be masked"

    edge_i, edge_w = build_soft_reply_graph(probs, threshold=0.0, valid_hist=valid)
    assert edge_i.size(0) == 2
    assert edge_i.size(1) == edge_w.numel()
    for j, t, w in zip(edge_i[0].tolist(), edge_i[1].tolist(), edge_w.tolist()):
        assert w == float(probs[t, j + 1])
        assert valid[t, j]
    print("predictor+graph ok", probs.round(decimals=3).tolist())


if __name__ == "__main__":
    test_predictor_and_graph()
    print("all smoke tests passed")
