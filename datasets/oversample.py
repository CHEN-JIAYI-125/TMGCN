#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train-only oversampling of rare (target, class) documents.

MT-CSD graphs are unique-sentence, so duplicating a JSON doc does not add
nodes. Extra copies still add `doc_final_*` rows. Training CE must run on
those last-sentence rows (`ce_on_doc_final`) for the copies to matter.

Trump is not a small target; Favor on Trump is the rare class. This balances
favor / against / none **inside each target** by resampling existing docs.
Dev and test must not be oversampled.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)

STANCE_NAMES = {0: "favor", 1: "against", 2: "none"}


def _doc_id(rec: Dict[str, Any]) -> int:
    return int(rec.get("doc_id", rec.get("id", -1)))


def _label(rec: Dict[str, Any]) -> int:
    return int(rec.get("label", -1))


def oversample_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_target: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        lab = _label(rec)
        if lab < 0:
            continue
        by_target[str(rec.get("target", ""))][lab] += 1
    out = {"docs": len(records), "targets": {}}
    for target, counts in sorted(by_target.items()):
        out["targets"][target] = {
            "favor": int(counts.get(0, 0)),
            "against": int(counts.get(1, 0)),
            "none": int(counts.get(2, 0)),
            "docs": int(sum(counts.values())),
        }
    return out


def oversample_records(
    records: Sequence[Dict[str, Any]],
    seed: int = 1234,
    max_factor: float = 10.0,
    classes: Iterable[int] = (0, 1, 2),
) -> List[Dict[str, Any]]:
    """Duplicate rare-class docs per target up to the target's largest class.

    Original rows are kept first. Copies get new ``doc_id`` / ``id`` values
    so dialogue hyperedges stay distinct; ``global_ids`` are reused so the
    frozen embedding table still hits.
    """
    class_set = {int(c) for c in classes}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get("target", ""))].append(rec)

    next_id = max((_doc_id(r) for r in records), default=0) + 1
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    cap = max(1.0, float(max_factor))

    for target in sorted(grouped):
        recs = grouped[target]
        out.extend(recs)
        by_cls: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for rec in recs:
            lab = _label(rec)
            if lab >= 0:
                by_cls[lab].append(rec)
        if not by_cls:
            continue
        peak = max(len(by_cls[c]) for c in by_cls)
        for lab in sorted(class_set):
            pool = by_cls.get(lab) or []
            if not pool:
                continue
            goal = min(peak, int(len(pool) * cap))
            need = goal - len(pool)
            if need <= 0:
                continue
            for _ in range(need):
                copy = deepcopy(pool[rng.randrange(len(pool))])
                copy["doc_id"] = next_id
                copy["id"] = next_id
                next_id += 1
                out.append(copy)
            logger.info(
                "oversample %s %s %d -> %d",
                target,
                STANCE_NAMES.get(lab, str(lab)),
                len(pool),
                len(pool) + need,
            )
    return out


def format_summary(title: str, summary: Dict[str, Any]) -> str:
    lines = ["%s docs=%d" % (title, int(summary["docs"]))]
    for target, row in summary.get("targets", {}).items():
        lines.append(
            "  %s | favor=%d against=%d none=%d docs=%d"
            % (
                target,
                int(row["favor"]),
                int(row["against"]),
                int(row["none"]),
                int(row["docs"]),
            )
        )
    return "\n".join(lines)


def oversample_json_file(
    src: Path,
    dst: Path,
    seed: int = 1234,
    max_factor: float = 10.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    recs = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(recs, list):
        raise ValueError("%s is not a JSON list" % src)
    before = oversample_summary(recs)
    out = oversample_records(recs, seed=seed, max_factor=max_factor)
    after = oversample_summary(out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return before, after


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="train_data.json")
    p.add_argument("--out", required=True, help="oversampled train JSON")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-factor", type=float, default=10.0)
    args = p.parse_args()
    before, after = oversample_json_file(
        Path(args.src), Path(args.out), seed=args.seed, max_factor=args.max_factor
    )
    print(format_summary("before", before), flush=True)
    print(format_summary("after", after), flush=True)
    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
