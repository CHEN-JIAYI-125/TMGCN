#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Student 软回复图。"""

from typing import Any

__all__ = [
    "ReplyPredictor",
    "build_soft_reply_graph",
    "masked_softmax",
    "apply_causal_reply_mask",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import reply_distillation

        return getattr(reply_distillation, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
