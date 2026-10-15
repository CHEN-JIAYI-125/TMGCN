#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""不加载大模型：检查 last_sentence 缓存行列对齐，以及不会把标签复制到所有话语。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from extract_conversation_embeddings import load_split_cache


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "cache" / "dummy_tiny"
    for split in ("train", "dev", "test"):
        emb, index_rows, config = load_split_cache(cache / split)
        assert emb.ndim == 2, emb.shape
        assert emb.shape[0] == len(index_rows)
        assert config.get("unit") == "last_sentence"
        ids = [row["conversation_id"] for row in index_rows]
        assert len(ids) == len(set(ids)), "last_sentence 不应把同一会话拆成多行同标签复制"
        turns = [row["n_turns"] for row in index_rows]
        last_turns = [row["turn_index"] for row in index_rows]
        for n_turns, last in zip(turns, last_turns):
            assert last == n_turns - 1
        print("OK", split, "n=%d dim=%d ids=%s" % (emb.shape[0], emb.shape[1], ids))
        np.testing.assert_equal(emb.dtype, np.float32)
    print("cache roundtrip passed:", cache)


if __name__ == "__main__":
    main()
