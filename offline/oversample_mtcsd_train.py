#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI wrapper: resample rare (target, class) docs in MT-CSD train JSON.

Training already oversamples in memory when yaml has oversample_minority: 1.
Use this only if you want a dumped JSON to inspect counts.

    python3.9 offline/oversample_mtcsd_train.py \\
        --src /home/chenjiayi/data/mtcsd/raw_with_global_ids/train_data.json \\
        --out /tmp/train_os.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.oversample import main

if __name__ == "__main__":
    main()
