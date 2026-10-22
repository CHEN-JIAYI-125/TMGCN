#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stance-label knowledge distillation: L = L_CE + λ T² KL(p_T || p_S).

This is independent of reply-structure distillation. The Teacher is a
generative LLM that outputs Favor / Against / Neutral probabilities; the
Student is the original TMGCN classifier logits. Original graph channels
are not modified.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

CLASS_NAMES = ("favor", "against", "neutral")
CLASS_ALIASES = {
    0: ("favor", "favour", "support", "支持", "赞成"),
    1: ("against", "oppose", "反对"),
    2: ("neutral", "none", "中立", "中性"),
}


def normalize_stance_probs(raw: Dict[str, Any], num_classes: int = 3) -> List[float]:
    """Map a Teacher dict onto [favor, against, neutral] and L1-normalize."""
    scores = [0.0] * num_classes
    src = raw.get("probs", raw) if isinstance(raw, dict) else {}
    if not isinstance(src, dict):
        src = {}
    for idx in range(num_classes):
        aliases = CLASS_ALIASES.get(idx, (CLASS_NAMES[idx],))
        val = 0.0
        for key in aliases:
            if key in src:
                try:
                    val = float(src[key])
                except (TypeError, ValueError):
                    val = 0.0
                break
        scores[idx] = max(0.0, val)
    total = sum(scores)
    if total <= 0.0:
        return [1.0 / num_classes] * num_classes
    return [v / total for v in scores]


def stance_kd_loss(
    student_logits,
    teacher_probs,
    teacher_mask,
    temperature: float = 3.0,
    smooth_teacher: bool = True,
):
    """T² KL(p_T || p_S) over Teacher-covered nodes.

    p_S = softmax(z_S / T). If ``smooth_teacher`` is True, p_T is also
    softened by p^{1/T} (Hinton KD). Missing Teacher rows contribute 0.
    """
    import torch
    import torch.nn.functional as F

    if student_logits is None:
        raise ValueError("student_logits is required")
    if teacher_probs is None or teacher_mask is None:
        return student_logits.new_zeros(())
    if teacher_probs.numel() == 0:
        return student_logits.new_zeros(())

    mask = teacher_mask.view(-1).to(dtype=torch.bool, device=student_logits.device)
    if not bool(mask.any()):
        return student_logits.new_zeros(())

    tau = max(float(temperature), 1e-6)
    log_p_s = F.log_softmax(student_logits[mask] / tau, dim=-1)
    p_t = teacher_probs.to(device=student_logits.device, dtype=student_logits.dtype)[mask]
    p_t = p_t.clamp_min(0.0)
    p_t = p_t / p_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if smooth_teacher:
        p_t = p_t.pow(1.0 / tau)
        p_t = p_t / p_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
    return (tau ** 2) * kl


def mock_teacher_probs(gold: int, num_classes: int = 3, smooth: float = 0.15) -> Sequence[float]:
    """Gold-smoothed dummy Teacher for pipeline tests. Not a real LLM."""
    if gold < 0 or gold >= num_classes:
        return [1.0 / num_classes] * num_classes
    remain = max(0.0, min(1.0, float(smooth)))
    peak = 1.0 - remain
    other = remain / max(1, num_classes - 1)
    probs = [other] * num_classes
    probs[gold] = peak
    return probs


def teacher_argmax(probs: Iterable[float]) -> int:
    values = [float(v) for v in probs]
    return int(max(range(len(values)), key=lambda i: values[i]))
