#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare TMGCN last-sentence preds with published SITPCL / GLAN.

Does not retrain those baselines. Published cells are from:
  SITPCL / GLAN on ZS-CSD: Ding et al., Findings of ACL 2025, Tables 4 and 6
  GLAN on MT-CSD: Niu et al., LREC-COLING 2024, Tables 6 and 7
SITPCL has no official MT-CSD numbers; that table omits it.

python3.9 scripts/analyze_vs_sitpcl_glan.py \\
  config/config_model5_rd_sem.yaml \\
  result/GraphTracer_5/<timestamp>/save/best_model.pt

python3.9 scripts/analyze_vs_sitpcl_glan.py \\
  config/config_model5_rd_sem_mtcsd.yaml \\
  result/GraphTracer_5_MT_CSD/20261118-143539/save/best_model.pt
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import target_graph_collate_fn  # noqa: E402
from main_graph import (  # noqa: E402
    _stance_block,
    _to_device,
    build_dataset,
    build_model,
    load_config,
    set_seed,
)

# Mixed / noun / claim: Favor, Against, Neutral, Macro-F1
ZSCSD_PUBLISHED = {
    "GLAN": {
        "mixed": (43.68, 34.91, 46.79, 41.79),
        "n": (45.91, 37.65, 51.58, 45.04),
        "c": (42.43, 33.18, 43.88, 39.83),
        "depth": (29.23, 36.21, 39.39, 38.11, 38.37, 48.93),
    },
    "SITPCL": {
        "mixed": (41.03, 41.34, 49.07, 43.81),
        "n": (40.99, 47.14, 54.49, 47.54),
        "c": (41.06, 37.34, 46.02, 41.47),
        "depth": (40.19, 38.37, 42.84, 40.62, 44.73, 45.25),
    },
}

# Per-target F_avg, then Avg.; depth F_avg per target in 1-2 / 3-5 / 6-8
MTCSD_GLAN = {
    "targets": ("Bitcoin", "Tesla", "SpaceX", "Biden", "Trump"),
    "favg": (56.95, 52.38, 55.98, 38.15, 48.91, 50.47),
    "depth": {
        "Bitcoin": (56.46, 59.76, 53.99),
        "Tesla": (24.44, 49.42, 54.92),
        "SpaceX": (50.77, 56.95, 53.23),
        "Biden": (28.95, 36.46, 42.01),
        "Trump": (33.33, 50.25, 47.35),
    },
}

ZSCSD_DEPTH_KEYS = ("1", "2", "3", "4", "5", ">=6")
MTCSD_DEPTH_KEYS = ("1-2", "3-5", "6-8")
MT_TARGETS = ("Biden", "Bitcoin", "SpaceX", "Tesla", "Trump")


def depth_from_turn(turn_index: int) -> int:
    """0-based last-sentence turn -> 1-based conversation depth."""
    return int(turn_index) + 1


def zscsd_depth_bin(turn_index: int) -> str:
    depth = depth_from_turn(turn_index)
    if depth >= 6:
        return ">=6"
    return str(depth)


def mtcsd_depth_bin(turn_index: int) -> str:
    depth = depth_from_turn(turn_index)
    if depth <= 2:
        return "1-2"
    if depth <= 5:
        return "3-5"
    return "6-8"


def _pct(x: float) -> str:
    return "%.2f" % (100.0 * float(x))


def _block_tuple(block: dict) -> Tuple[float, float, float, float, float]:
    return (
        100.0 * float(block["f1_favor"]),
        100.0 * float(block["f1_against"]),
        100.0 * float(block["f1_neutral"]),
        100.0 * float(block["f1_macro"]),
        100.0 * float(block["f1_favg"]),
    )


def collect_last_sentence_rows(model, dataset, device) -> List[dict]:
    model.eval()
    rows: List[dict] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            batch = _to_device(target_graph_collate_fn([item]), device)
            logits = model(**batch)["logits"]
            preds = logits.argmax(dim=-1).cpu().tolist()
            target_text = dataset.target_texts[idx]
            target_type = getattr(dataset, "target_types", [""] * len(dataset))[idx]
            utterance_ids = batch["utterance_ids"].cpu().tolist()
            speaker_ids = batch["speaker_ids"].cpu().tolist()
            final_docs = batch["doc_final_doc_ids"].cpu().tolist()
            final_nodes = batch["doc_final_node_index"].cpu().tolist()
            final_turns = batch["doc_final_turn_index"].cpu().tolist()
            final_true = batch["doc_final_true"].cpu().tolist()
            for doc, node, turn, true in zip(
                final_docs, final_nodes, final_turns, final_true
            ):
                if int(true) < 0:
                    continue
                rows.append(
                    {
                        "doc_id": doc,
                        "target": target_text,
                        "target_type": str(target_type),
                        "utterance_id": utterance_ids[node],
                        "sentence_idx": int(turn),
                        "speaker_id": speaker_ids[node],
                        "true": int(true),
                        "pred": int(preds[node]),
                    }
                )
    return rows


def subset_block(rows: Sequence[dict], num_classes: int = 3) -> dict:
    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    return _stance_block(y_true, y_pred, num_classes)


def grouped_blocks(rows: Sequence[dict], key_fn) -> Dict[str, dict]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    return {k: subset_block(v) for k, v in buckets.items()}


def is_mtcsd(rows: Sequence[dict]) -> bool:
    names = {str(r["target"]) for r in rows}
    return bool(names.intersection(MT_TARGETS))


def _fmt_row(name: str, cells: Sequence[float], width: int = 6) -> str:
    body = " & ".join("%*.2f" % (width, float(x)) for x in cells)
    return "%s & %s \\\\" % (name, body)


def print_zscsd(rows: List[dict]) -> None:
    mixed = subset_block(rows)
    by_type = grouped_blocks(rows, lambda r: r["target_type"])
    noun = by_type.get("n", _stance_block([], [], 3))
    claim = by_type.get("c", _stance_block([], [], 3))
    tmgcn = {
        "mixed": _block_tuple(mixed)[:4],
        "n": _block_tuple(noun)[:4],
        "c": _block_tuple(claim)[:4],
    }
    by_depth = grouped_blocks(rows, lambda r: zscsd_depth_bin(r["sentence_idx"]))
    tmgcn_depth = tuple(
        100.0 * float(by_depth.get(k, _stance_block([], [], 3))["f1_macro"])
        for k in ZSCSD_DEPTH_KEYS
    )

    print("\n=== ZS-CSD vs GLAN / SITPCL (Macro-F1, last sentence) ===", flush=True)
    print(
        "Method | Mixed Fav/Ag/Neu/All | Noun Fav/Ag/Neu/All | Claim Fav/Ag/Neu/All",
        flush=True,
    )
    for name, pack in ZSCSD_PUBLISHED.items():
        print(
            "%s | %s | %s | %s"
            % (
                name,
                " ".join("%.2f" % x for x in pack["mixed"]),
                " ".join("%.2f" % x for x in pack["n"]),
                " ".join("%.2f" % x for x in pack["c"]),
            ),
            flush=True,
        )
    print(
        "TMGCN | %s | %s | %s"
        % (
            " ".join("%.2f" % x for x in tmgcn["mixed"]),
            " ".join("%.2f" % x for x in tmgcn["n"]),
            " ".join("%.2f" % x for x in tmgcn["c"]),
        ),
        flush=True,
    )
    print("n: mixed=%d noun=%d claim=%d" % (mixed["num"], noun["num"], claim["num"]), flush=True)

    print("\n=== ZS-CSD conversation depth (Macro-F1) ===", flush=True)
    print("Method | " + " | ".join(ZSCSD_DEPTH_KEYS), flush=True)
    for name, pack in ZSCSD_PUBLISHED.items():
        print("%s | %s" % (name, " | ".join("%.2f" % x for x in pack["depth"])), flush=True)
    print("TMGCN | %s" % " | ".join("%.2f" % x for x in tmgcn_depth), flush=True)
    for k in ZSCSD_DEPTH_KEYS:
        print("  depth %s n=%d" % (k, by_depth.get(k, {"num": 0})["num"]), flush=True)

    print("\n% ZS-CSD LaTeX (Favor / Against / Neutral / Macro-F1)", flush=True)
    print("GLAN & " + " & ".join("%.2f" % x for x in ZSCSD_PUBLISHED["GLAN"]["mixed"] + ZSCSD_PUBLISHED["GLAN"]["n"] + ZSCSD_PUBLISHED["GLAN"]["c"]) + " \\\\", flush=True)
    print("SITPCL & " + " & ".join("%.2f" % x for x in ZSCSD_PUBLISHED["SITPCL"]["mixed"] + ZSCSD_PUBLISHED["SITPCL"]["n"] + ZSCSD_PUBLISHED["SITPCL"]["c"]) + " \\\\", flush=True)
    print("TMGCN & " + " & ".join("%.2f" % x for x in tmgcn["mixed"] + tmgcn["n"] + tmgcn["c"]) + " \\\\", flush=True)
    print("% depth Macro-F1", flush=True)
    print(_fmt_row("GLAN", ZSCSD_PUBLISHED["GLAN"]["depth"]), flush=True)
    print(_fmt_row("SITPCL", ZSCSD_PUBLISHED["SITPCL"]["depth"]), flush=True)
    print(_fmt_row("TMGCN", tmgcn_depth), flush=True)


def print_mtcsd(rows: List[dict]) -> None:
    by_target = grouped_blocks(rows, lambda r: str(r["target"]))
    tmgcn_favg = []
    print("\n=== MT-CSD vs GLAN (F_avg). SITPCL has no official MT-CSD result ===", flush=True)
    print("Method | Bitcoin Tesla SpaceX Biden Trump Avg.", flush=True)
    print(
        "GLAN | %s"
        % " ".join("%.2f" % x for x in MTCSD_GLAN["favg"]),
        flush=True,
    )
    cells = []
    for name in MTCSD_GLAN["targets"]:
        block = by_target.get(name, _stance_block([], [], 3))
        cells.append(100.0 * float(block["f1_favg"]))
        tmgcn_favg.append(100.0 * float(block["f1_favg"]))
        print(
            "  %s n=%d Favor=%s Against=%s None=%s F_avg=%s Macro=%s"
            % (
                name,
                block["num"],
                _pct(block["f1_favor"]),
                _pct(block["f1_against"]),
                _pct(block["f1_neutral"]),
                _pct(block["f1_favg"]),
                _pct(block["f1_macro"]),
            ),
            flush=True,
        )
    avg = float(np.mean(cells)) if cells else 0.0
    print("TMGCN | %s %.2f" % (" ".join("%.2f" % x for x in cells), avg), flush=True)

    print("\n=== MT-CSD depth F_avg (1-2 / 3-5 / 6-8), same bins as GLAN Table 7 ===", flush=True)
    print("Target | GLAN 1-2 3-5 6-8 | TMGCN 1-2 3-5 6-8 | n", flush=True)
    latex_glan = []
    latex_tmgcn = []
    for name in MTCSD_GLAN["targets"]:
        sub = [r for r in rows if str(r["target"]) == name]
        by_d = grouped_blocks(sub, lambda r: mtcsd_depth_bin(r["sentence_idx"]))
        ours = tuple(
            100.0 * float(by_d.get(k, _stance_block([], [], 3))["f1_favg"])
            for k in MTCSD_DEPTH_KEYS
        )
        ns = tuple(int(by_d.get(k, {"num": 0})["num"]) for k in MTCSD_DEPTH_KEYS)
        glan = MTCSD_GLAN["depth"][name]
        print(
            "%s | %.2f %.2f %.2f | %.2f %.2f %.2f | %s"
            % (name, glan[0], glan[1], glan[2], ours[0], ours[1], ours[2], " ".join(str(x) for x in ns)),
            flush=True,
        )
        latex_glan.extend(glan)
        latex_tmgcn.extend(ours)

    print("\n=== MT-CSD confusion (rows=gold Favor/Against/None, cols=pred) ===", flush=True)
    for name in MTCSD_GLAN["targets"]:
        sub = [r for r in rows if str(r["target"]) == name]
        if not sub:
            continue
        mat = confusion_matrix(
            [r["true"] for r in sub],
            [r["pred"] for r in sub],
            labels=[0, 1, 2],
        )
        gold_fav = int(mat[0].sum())
        pred_fav = int(mat[:, 0].sum())
        print(
            "%s\n%s  gold_favor=%d pred_favor=%d tp_favor=%d"
            % (name, mat, gold_fav, pred_fav, int(mat[0, 0])),
            flush=True,
        )

    print("\n% MT-CSD LaTeX F_avg", flush=True)
    print(_fmt_row("GLAN", MTCSD_GLAN["favg"]), flush=True)
    print(_fmt_row("TMGCN", list(cells) + [avg]), flush=True)
    print("% depth F_avg Bitcoin/Tesla/SpaceX/Biden/Trump x (1-2,3-5,6-8)", flush=True)
    print(_fmt_row("GLAN", latex_glan), flush=True)
    print(_fmt_row("TMGCN", latex_tmgcn), flush=True)


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "usage: python3.9 scripts/analyze_vs_sitpcl_glan.py <config.yaml> <best_model.pt>",
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
    rows = collect_last_sentence_rows(model, test_dataset, device)
    print("last-sentence rows: %d" % len(rows), flush=True)
    if is_mtcsd(rows):
        print_mtcsd(rows)
    else:
        print_zscsd(rows)


if __name__ == "__main__":
    main()
