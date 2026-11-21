#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count TMGCN training-graph parameters. No data, no GPU required.

Qwen3-Embedding-8B is encoded offline and is NOT part of this module.
Report it separately as a frozen encoder, not as trainable GNN weights.

    python3.9 scripts/count_complexity.py config/config_model5_rd_sem.yaml --input-size 4096
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_graph import build_model, count_params, load_config


# Public Qwen3-8B / Qwen3-Embedding-8B scale (not loaded at train time).
QWEN3_EMBEDDING_8B = 8_190_000_000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="yaml, e.g. config/config_model5_rd_sem.yaml")
    p.add_argument(
        "--input-size",
        type=int,
        default=4096,
        help="frozen embedding width (Qwen3-Embedding-8B is 4096)",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg.input_size = int(args.input_size)
    model = build_model(cfg)
    stats = count_params(model)
    out = {
        "config": str(args.config),
        "input_size": cfg.input_size,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "reply_distillation": int(getattr(cfg, "reply_distillation", 0) or 0),
        "training_graph": stats,
        "encoder": {
            "name": "Qwen3-Embedding-8B",
            "approx_params": QWEN3_EMBEDDING_8B,
            "frozen": True,
            "used_at": "offline encode only; not in the training module",
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(
        "GNN trainable=%.2fM | GNN total=%.2fM | encoder frozen ~%.2fB (offline)"
        % (
            stats["trainable_params"] / 1e6,
            stats["total_params"] / 1e6,
            QWEN3_EMBEDDING_8B / 1e9,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
