#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-evaluate an MT-CSD checkpoint. No retraining.

Official nfq729/MT-CSD metric: F_avg = (F1_favor + F1_against) / 2.
Our ids: 0=favor, 1=against, 2=none. Also prints mixed/c/n.

python3.9 scripts/eval_mtcsd.py \\
  config/config_model5_rd_sem_mtcsd.yaml \\
  result/GraphTracer_5_MT_CSD/<timestamp>/save/best_model.pt
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


def _pct(x) -> str:
    return "%.2f" % (100.0 * float(x))


def _line(title: str, metrics: dict, key: str) -> str:
    return (
        "%s | Macro F1: %s | F_avg: %s | Favor: %s | Against: %s | Neutral/None: %s | n=%d"
        % (
            title,
            _pct(metrics["%s_f1_macro" % key]),
            _pct(metrics["%s_f1_favg" % key]),
            _pct(metrics["%s_f1_favor" % key]),
            _pct(metrics["%s_f1_against" % key]),
            _pct(metrics["%s_f1_neutral" % key]),
            int(metrics["%s_num" % key]),
        )
    )


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "usage: python3.9 scripts/eval_mtcsd.py <config.yaml> <best_model.pt>",
            flush=True,
        )
        sys.exit(2)
    cfg = load_config(sys.argv[1])
    set_seed(cfg.seed)
    ckpt = torch.load(sys.argv[2], map_location="cpu")
    device = torch.device(
        "cuda:%d" % cfg.cuda_index if torch.cuda.is_available() else "cpu"
    )
    test_dataset = build_dataset(cfg, cfg.test_path)
    cfg.input_size = test_dataset.embedding_store.dim
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    metrics = predict_and_evaluate(
        model, test_dataset, cfg, device, prefix="test", pred_path=None
    )
    print("", flush=True)
    print("=== TMGCN / ZS-CSD scopes (claim empty on MT-CSD) ===", flush=True)
    print(_line("mixed", metrics, "mixed"), flush=True)
    print(_line("claim(c)", metrics, "c"), flush=True)
    print(_line("noun(n)", metrics, "n"), flush=True)
    print("", flush=True)
    print("=== official MT-CSD last row: F_avg per target ===", flush=True)
    names = metrics.get("per_target_names") or []
    cells = []
    for name in names:
        print(_line(name, metrics, "t_%s" % name), flush=True)
        cells.append("%s %s" % (name, _pct(metrics["t_%s_f1_favg" % name])))
    if names:
        print(
            "TMGCN last row | %s | Overall F_avg %s | mixed Macro F1 %s"
            % (
                " | ".join(cells),
                _pct(metrics["mixed_f1_favg"]),
                _pct(metrics["mixed_f1_macro"]),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
