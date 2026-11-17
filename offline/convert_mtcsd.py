#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert official MT-CSD (nfq729/MT-CSD) into TMGCN conversation JSON.

MT-CSD layout (cloned repo ``data/``)::

    data/{Biden,Bitcoin,SpaceX,Tesla,Trump}/
        text.csv       first column = comment id, column ``text``
        train.json     list of {index: [id, ...], stance: favor|against|none}
        valid.json
        test.json

Each JSON item is one annotated comment: ``index[-1]`` is the comment,
the prefix is its ancestor path. That matches TMGCN last-sentence scoring
(``metric_on_last_sentence=true``): one doc per annotated instance.

Stance mapping (TMGCN): favor=0, against=1, none=2.
Official MT-CSD F_avg is (F1_favor + F1_against) / 2; the trainer logs it
as mixed F_avg. Do not put gold labels into embedding prompts.

    python3.9 offline/convert_mtcsd.py \\
        --src ~/MT-CSD/data \\
        --out /home/chenjiayi/data/mtcsd/raw_with_global_ids
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


TARGETS = ("Biden", "Bitcoin", "SpaceX", "Tesla", "Trump")
SPLITS = (("train", "train.json"), ("dev", "valid.json"), ("test", "test.json"))
STANCE_TO_ID = {"favor": 0, "against": 1, "none": 2}


def _norm_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            if float(text) == int(float(text)):
                text = str(int(float(text)))
        except ValueError:
            pass
    return text


def load_text_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("empty CSV header: %s" % path)
        id_col = reader.fieldnames[0]
        if "text" not in reader.fieldnames:
            raise ValueError("%s has no 'text' column" % path)
        rows = {}
        for row in reader:
            cid = _norm_id(row.get(id_col, ""))
            if not cid:
                continue
            rows[cid] = str(row.get("text") or "")
    if not rows:
        raise ValueError("no comments in %s" % path)
    return rows


def load_split_items(path: Path) -> List[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("%s is not a JSON list" % path)
    return items


def stance_id(raw: str) -> int:
    key = str(raw or "").strip().lower()
    if key not in STANCE_TO_ID:
        raise ValueError("unknown stance %r" % raw)
    return STANCE_TO_ID[key]


def convert_target(
    target: str,
    text_map: Dict[str, str],
    items: List[dict],
    split: str,
    gid_table: Dict[Tuple[str, str], int],
    speaker_table: Dict[Tuple[str, str], int],
    next_gid: List[int],
    next_speaker: List[int],
    next_doc: List[int],
) -> List[dict]:
    last_label = {}
    for item in items:
        ids = [_norm_id(x) for x in item.get("index") or []]
        if not ids:
            continue
        last_label[ids[-1]] = stance_id(item.get("stance"))

    records = []
    missing = 0
    for item in items:
        ids = [_norm_id(x) for x in item.get("index") or []]
        if not ids:
            continue
        sentences = []
        global_ids = []
        speakers = []
        all_labels = []
        skip = False
        for cid in ids:
            if cid not in text_map:
                missing += 1
                skip = True
                break
            key = (target, cid)
            if key not in gid_table:
                gid_table[key] = next_gid[0]
                next_gid[0] += 1
            if key not in speaker_table:
                speaker_table[key] = next_speaker[0]
                next_speaker[0] += 1
            sentences.append(text_map[cid])
            global_ids.append(gid_table[key])
            speakers.append(speaker_table[key])
            all_labels.append(last_label.get(cid, -100))
        if skip:
            continue
        last = all_labels[-1]
        if last < 0:
            last = stance_id(item.get("stance"))
            all_labels[-1] = last
        doc_id = next_doc[0]
        next_doc[0] += 1
        records.append(
            {
                "id": doc_id,
                "target": target,
                "target_type": "n",
                "label": int(last),
                "speakers": speakers,
                "sentences": sentences,
                "all_labels": all_labels,
                "global_ids": global_ids,
                "mtcsd_index": ids,
                "mtcsd_split": split,
            }
        )
    if missing:
        raise ValueError("%s/%s missing %d comment texts" % (target, split, missing))
    return records


def write_summary(out_dir: Path, splits: Dict[str, List[dict]]) -> None:
    summary = {"targets": list(TARGETS), "splits": {}}
    for name, recs in splits.items():
        labels = Counter(int(r["label"]) for r in recs)
        targets = Counter(str(r["target"]) for r in recs)
        summary["splits"][name] = {
            "docs": len(recs),
            "by_target": dict(targets),
            "labels": {
                "favor": int(labels.get(0, 0)),
                "against": int(labels.get(1, 0)),
                "none": int(labels.get(2, 0)),
            },
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="MT-CSD data/ directory")
    p.add_argument("--out", required=True, help="output directory for split JSON")
    args = p.parse_args()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)

    gid_table: Dict[Tuple[str, str], int] = {}
    speaker_table: Dict[Tuple[str, str], int] = {}
    next_gid = [0]
    next_speaker = [0]
    next_doc = [1]
    splits: Dict[str, List[dict]] = {name: [] for name, _ in SPLITS}

    for target in TARGETS:
        folder = src / target
        text_path = folder / "text.csv"
        if not text_path.is_file():
            raise FileNotFoundError(text_path)
        text_map = load_text_csv(text_path)
        for split, fname in SPLITS:
            items = load_split_items(folder / fname)
            splits[split].extend(
                convert_target(
                    target,
                    text_map,
                    items,
                    split,
                    gid_table,
                    speaker_table,
                    next_gid,
                    next_speaker,
                    next_doc,
                )
            )

    out.mkdir(parents=True, exist_ok=True)
    for split, recs in splits.items():
        path = out / ("%s_data.json" % split)
        path.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        print("%s %s docs=%d" % (path, split, len(recs)), flush=True)
    write_summary(out, splits)
    print(
        "COMPLETE unique comments=%d speakers=%d docs=%d"
        % (len(gid_table), len(speaker_table), next_doc[0] - 1),
        flush=True,
    )


if __name__ == "__main__":
    main()
