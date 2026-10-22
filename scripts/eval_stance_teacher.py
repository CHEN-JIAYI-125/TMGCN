#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect Stance Teacher quality against gold last-sentence labels.

Official CSD metric: last sentence of each doc, mixed / claim(c) / noun(n)
Macro F1, class order 0=favor, 1=against, 2=neutral.

    python scripts/eval_stance_teacher.py --teacher_path data/stance_teacher
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_stance_teacher import evaluate_teacher_dir  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate offline stance Teacher JSON")
    p.add_argument("--teacher_path", default="data/stance_teacher")
    args = p.parse_args()
    root = Path(args.teacher_path)
    if not root.exists():
        raise FileNotFoundError("teacher path does not exist: %s" % root)
    metrics = evaluate_teacher_dir(root)
    print(
        "Teacher last-sentence | Macro F1: %.2f | Favor: %.2f | Against: %.2f | Neutral: %.2f | n=%d"
        % (
            100.0 * metrics.get("mixed_f1_macro", 0.0),
            100.0 * metrics.get("mixed_f1_favor", 0.0),
            100.0 * metrics.get("mixed_f1_against", 0.0),
            100.0 * metrics.get("mixed_f1_neutral", 0.0),
            metrics.get("mixed_num", 0),
        )
    )
    print(
        "claim(c) | Macro F1: %.2f | n=%d || noun(n) | Macro F1: %.2f | n=%d"
        % (
            100.0 * metrics.get("c_f1_macro", 0.0),
            metrics.get("c_num", 0),
            100.0 * metrics.get("n_f1_macro", 0.0),
            metrics.get("n_num", 0),
        )
    )
    out = root / "teacher_quality.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
