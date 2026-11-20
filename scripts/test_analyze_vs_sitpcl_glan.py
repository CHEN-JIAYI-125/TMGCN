#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_vs_sitpcl_glan import (  # noqa: E402
    depth_from_turn,
    mtcsd_depth_bin,
    zscsd_depth_bin,
)


def test_depth_bins() -> None:
    assert depth_from_turn(0) == 1
    assert zscsd_depth_bin(0) == "1"
    assert zscsd_depth_bin(4) == "5"
    assert zscsd_depth_bin(5) == ">=6"
    assert mtcsd_depth_bin(0) == "1-2"
    assert mtcsd_depth_bin(1) == "1-2"
    assert mtcsd_depth_bin(2) == "3-5"
    assert mtcsd_depth_bin(4) == "3-5"
    assert mtcsd_depth_bin(5) == "6-8"
    assert mtcsd_depth_bin(7) == "6-8"


if __name__ == "__main__":
    test_depth_bins()
    print("ok")
