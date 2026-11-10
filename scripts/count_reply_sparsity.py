#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load a trained checkpoint and report test Mixed F1 plus avg reply edges/node.

python3.9 scripts/count_reply_sparsity.py \\
    config/config_model5_rd_sem.yaml \\
    result/GraphTracer_5/<timestamp>/save/best_model.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_graph import (  # noqa: E402
    build_dataset,
    build_model,
    load_config,
    predict_and_evaluate,
    set_seed,
)


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "usage: python3.9 scripts/count_reply_sparsity.py "
            "<config.yaml> <best_model.pt>",
            flush=True,
        )
        sys.exit(2)
    cfg = load_config(sys.argv[1])
    set_seed(cfg.seed)
    ckpt_path = Path(sys.argv[2])
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    device = torch.device(
        f"cuda:{cfg.cuda_index}" if torch.cuda.is_available() else "cpu"
    )
    test_dataset = build_dataset(cfg, cfg.test_path)
    cfg.input_size = test_dataset.embedding_store.dim
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    metrics = predict_and_evaluate(
        model, test_dataset, cfg, device, prefix="test", pred_path=None
    )
    avg = metrics.get("reply_avg_edges_per_node")
    print(
        "threshold=%.3f | Mixed Macro-F1=%.2f | avg_edges/node=%s | edges=%s | nodes=%s"
        % (
            float(cfg.reply_threshold),
            100.0 * float(metrics["mixed_f1_macro"]),
            ("%.4f" % avg) if avg is not None else "--",
            metrics.get("reply_num_edges", "--"),
            metrics.get("reply_num_nodes", "--"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
