#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Claim–noun polarity helpers.

CSD claim targets are often a negated / opposing proposition about a noun
target, e.g. 「反对日本动漫文化」 vs 「日本动漫文化」, 「sora没什么用处」 vs
「Sora」. Favoring the claim is then closer to *against* the noun. These
helpers only attach matching metadata; they do not change graph topology.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

NEG_PREFIXES = (
    "反对", "取消", "禁止", "拒绝", "废除", "消除", "不要", "不得",
    "不该", "不应该", "停止", "建议消除", "像国外一样取消",
)
NEG_INFIXES = ("没什么", "没有用", "无用处", "没用处")


def claim_is_negated(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if any(t.startswith(p) for p in NEG_PREFIXES):
        return True
    return any(p in t for p in NEG_INFIXES)


def load_noun_targets(template_path: Optional[str] = None) -> List[str]:
    path = Path(template_path) if template_path else Path(__file__).with_name("target_knowledge_template.json")
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    nouns = []
    for target, info in data.items():
        if str((info or {}).get("target_type", "")).lower() == "n":
            nouns.append(str(target))
    return nouns


def related_noun(claim: str, noun_texts: Iterable[str]) -> Optional[str]:
    """Longest noun-target string contained in the claim (case-insensitive)."""
    hay = (claim or "").casefold()
    best = ""
    best_orig = None
    for raw in noun_texts:
        n = (raw or "").strip()
        if len(n) < 2:
            continue
        if n.casefold() in hay and len(n) > len(best):
            best = n
            best_orig = raw
    return best_orig


def attach_claim_noun_fields(
    graphs: List[Any],
    knowledge_store: Optional[Any],
    template_path: Optional[str] = None,
) -> Tuple[int, int]:
    nouns = load_noun_targets(template_path)
    if knowledge_store is not None:
        for key in knowledge_store.index:
            if key not in nouns:
                nouns.append(key)
    n_claim = 0
    n_paired = 0
    n_neg = 0
    dim = knowledge_store.dim if knowledge_store is not None else 0
    for graph in graphs:
        typ = str(getattr(graph, "target_type", "") or "")
        graph.claim_negated = 0
        graph.related_noun_text = ""
        if dim:
            import torch
            graph.related_noun_knowledge = torch.zeros((4, dim), dtype=torch.float32)
        if typ != "c":
            continue
        n_claim += 1
        if claim_is_negated(graph.target_text):
            graph.claim_negated = 1
            n_neg += 1
        noun = related_noun(graph.target_text, nouns)
        if noun:
            graph.related_noun_text = noun
            if knowledge_store is not None:
                know = knowledge_store.get(noun)
                if know.abs().sum() > 0:
                    graph.related_noun_knowledge = know
                    n_paired += 1
    attach_opp_labels(graphs)
    logger.info(
        "claim–noun polarity | claims=%d negated=%d paired_noun=%d",
        n_claim, n_neg, n_paired,
    )
    return n_paired, n_claim


def attach_opp_labels(graphs: List[Any]) -> int:
    """For negated claims, extra train target = flipped gold of the same utterance on the related noun.

    Favor(0) <-> Against(1), Neutral stays 2. Unpaired nodes stay -100 (ignored).
    """
    import torch

    noun_lab: Dict[Tuple[str, int], int] = {}
    for graph in graphs:
        if str(getattr(graph, "target_type", "") or "") != "n":
            continue
        for gid, lab in zip(graph.utterance_ids.tolist(), graph.y.tolist()):
            if int(lab) >= 0:
                noun_lab[(graph.target_text, int(gid))] = int(lab)
    n_opp = 0
    for graph in graphs:
        n = graph.num_nodes
        opp = torch.full((n,), -100, dtype=torch.long)
        if (
            str(getattr(graph, "target_type", "") or "") == "c"
            and int(getattr(graph, "claim_negated", 0)) == 1
            and getattr(graph, "related_noun_text", "")
        ):
            noun = graph.related_noun_text
            for i, gid in enumerate(graph.utterance_ids.tolist()):
                lab = noun_lab.get((noun, int(gid)))
                if lab is None:
                    continue
                if lab in (0, 1):
                    opp[i] = 1 - lab
                    n_opp += 1
                elif lab == 2:
                    opp[i] = 2
                    n_opp += 1
        graph.opp_y = opp
    logger.info("opposite aux labels on %d claim nodes", n_opp)
    return n_opp
