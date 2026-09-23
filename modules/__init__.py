#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TMGCN 扩展模块包。Reply Structure Knowledge Distillation 位于 reply_distillation。"""

from .reply_distillation import (
    ReplyKDLoss,
    ReplyPredictor,
    apply_causal_reply_mask,
    build_soft_reply_graph,
    masked_softmax,
)

__all__ = [
    "ReplyPredictor",
    "ReplyKDLoss",
    "build_soft_reply_graph",
    "masked_softmax",
    "apply_causal_reply_mask",
]
