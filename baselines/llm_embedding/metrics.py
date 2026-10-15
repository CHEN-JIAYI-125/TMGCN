#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""与 main_graph.py 相同口径的 CSD 指标（独立复制，不改 TMGCN）。

标签约定：0=favor, 1=against, 2=neutral。
scope: mixed=全部, c=claim target, n=noun-phrase target。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _class_f1s(y_true, y_pred, num_classes: int) -> List[float]:
    per_class = f1_score(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0.0
    )
    return [float(v) for v in per_class]


def scoped_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_types: List[str],
    num_classes: int = 3,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    types_arr = np.asarray([str(t) for t in target_types])
    scopes = {
        "mixed": np.ones(types_arr.shape, dtype=bool),
        "c": types_arr == "c",
        "n": types_arr == "n",
    }
    metrics: Dict[str, Any] = {}
    for scope_name, mask in scopes.items():
        n_scored = int(mask.sum())
        if n_scored == 0:
            metrics.update(
                {
                    "%s_f1_macro" % scope_name: 0.0,
                    "%s_f1_favor" % scope_name: 0.0,
                    "%s_f1_against" % scope_name: 0.0,
                    "%s_f1_neutral" % scope_name: 0.0,
                    "%s_accuracy" % scope_name: 0.0,
                    "%s_num" % scope_name: 0,
                }
            )
            continue
        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]
        f1_per_class = _class_f1s(yt, yp, num_classes)
        favor = f1_per_class[0] if num_classes > 0 else 0.0
        against = f1_per_class[1] if num_classes > 1 else 0.0
        neutral = f1_per_class[2] if num_classes > 2 else 0.0
        metrics.update(
            {
                "%s_f1_macro" % scope_name: float(
                    f1_score(yt, yp, average="macro", zero_division=0.0)
                ),
                "%s_f1_favor" % scope_name: float(favor),
                "%s_f1_against" % scope_name: float(against),
                "%s_f1_neutral" % scope_name: float(neutral),
                "%s_accuracy" % scope_name: float(accuracy_score(yt, yp)),
                "%s_num" % scope_name: n_scored,
            }
        )
    return metrics


def format_split_line(title: str, loss: float, metrics: Dict[str, Any]) -> str:
    def pct(x: float) -> str:
        return "%.2f" % (float(x) * 100.0)

    return (
        "%s | Loss: %.2f | Macro F1: %s | Favor: %s | Against: %s | Neutral: %s"
        % (
            title,
            float(loss),
            pct(metrics["mixed_f1_macro"]),
            pct(metrics["mixed_f1_favor"]),
            pct(metrics["mixed_f1_against"]),
            pct(metrics["mixed_f1_neutral"]),
        )
    )
