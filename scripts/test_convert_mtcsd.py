#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tiny fixture test for offline/convert_mtcsd.py."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.convert_mtcsd import TARGETS, convert_target, load_text_csv, main


def test_path_to_last_sentence_doc(tmp_path: Path) -> None:
    folder = tmp_path / "Biden"
    folder.mkdir()
    with (folder / "text.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerow({"id": "1", "text": "Biden gave a speech."})
        writer.writerow({"id": "1-1", "text": "I disagree with that policy."})
    items = [
        {"index": ["1"], "stance": "none"},
        {"index": ["1", "1-1"], "stance": "against"},
    ]
    (folder / "train.json").write_text(json.dumps(items), encoding="utf-8")
    (folder / "valid.json").write_text("[]", encoding="utf-8")
    (folder / "test.json").write_text("[]", encoding="utf-8")
    for name in TARGETS:
        if name == "Biden":
            continue
        other = tmp_path / name
        other.mkdir()
        with (other / "text.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "text"])
            writer.writeheader()
            writer.writerow({"id": "x", "text": "placeholder"})
        empty = [{"index": ["x"], "stance": "none"}]
        (other / "train.json").write_text("[]", encoding="utf-8")
        (other / "valid.json").write_text("[]", encoding="utf-8")
        (other / "test.json").write_text(json.dumps(empty), encoding="utf-8")

    out = tmp_path / "out"
    sys.argv = ["convert_mtcsd.py", "--src", str(tmp_path), "--out", str(out)]
    main()
    train = json.loads((out / "train_data.json").read_text(encoding="utf-8"))
    assert len(train) == 2
    short, long = train
    assert short["sentences"] == ["Biden gave a speech."]
    assert short["label"] == 2  # none
    assert long["sentences"][-1] == "I disagree with that policy."
    assert long["label"] == 1  # against
    assert long["all_labels"][-1] == 1
    assert long["global_ids"][0] == short["global_ids"][0]


def test_load_csv_string_ids(tmp_path: Path) -> None:
    path = tmp_path / "text.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerow({"id": "1-1-2", "text": "hello"})
    rows = load_text_csv(path)
    recs = convert_target(
        "Tesla",
        rows,
        [{"index": ["1-1-2"], "stance": "favor"}],
        "train",
        {},
        {},
        [0],
        [0],
        [1],
    )
    assert recs[0]["label"] == 0
    assert recs[0]["target_type"] == "n"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        test_load_csv_string_ids(tmp)
        test_path_to_last_sentence_doc(tmp)
    print("convert_mtcsd fixture tests passed")
