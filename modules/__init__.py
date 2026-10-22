#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TMGCN 扩展模块包。Reply Structure KD 与 Stance-label KD 分开。

子模块按需导入，避免 ``from modules.stance_distillation import ...``
时把 Reply 蒸馏的 torch 依赖一并拉起来。
"""

from typing import Any

__all__ = [
    "ReplyPredictor",
    "ReplyKDLoss",
    "build_soft_reply_graph",
    "masked_softmax",
    "apply_causal_reply_mask",
    "stance_kd_loss",
]


def __getattr__(name: str) -> Any:
    if name == "stance_kd_loss":
        from .stance_distillation import stance_kd_loss

        return stance_kd_loss
    if name in {
        "ReplyPredictor",
        "ReplyKDLoss",
        "build_soft_reply_graph",
        "masked_softmax",
        "apply_causal_reply_mask",
    }:
        from . import reply_distillation

        return getattr(reply_distillation, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
