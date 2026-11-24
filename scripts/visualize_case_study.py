#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paper case-study figures: keep every in-doc node, shorten text only.

Layout (one thread, top to bottom):
  Dialogue bracket on the left of the stack.
  Speaker-history dashes further left (same speaker, may skip turns).
  Temporal solids along the node spine (consecutive turns only).
  Reply curves on the right from Soft Reply P(j|t), never from adjacency.

Shortening never deletes a node. Non-key turns become gray chips that still
carry the original utterance index and speaker id, so a Reply arrow whose
antecedent is not expanded still lands on that placeholder.

    python3.9 scripts/visualize_case_study.py
    python3.9 scripts/visualize_case_study.py \\
        --input results/case_study_analysis/four_cases.json \\
        --out results/case_study_analysis
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results" / "case_study_analysis"

# Distinct by both hue and dash. Do not reuse Temporal color for Reply.
COL_TEMPORAL = "#1f6f8b"
COL_SPEAKER = "#c45c26"
COL_REPLY = "#5b3a91"
COL_DIALOGUE = "#2f6f4e"
COL_COMPACT = "#d9d9d9"
COL_COMPACT_STROKE = "#8a8a8a"
COL_EXPAND = "#ffffff"
COL_EXPAND_STROKE = "#333333"
COL_LAST = "#111111"
COL_TEXT = "#222222"
COL_MUTED = "#666666"

# Paper threads. Identity / stance / reply targets are overwritten by extract.
# text_en is display-only and does not change speaker_id or utterance index.
PAPER_CATALOG: Dict[int, Dict[str, Any]] = {
    681: {
        "case_id": 1,
        "role": "Cross-speaker thread",
        "title": "Case 1 — Cross-speaker thread",
        "target_en": "A man livestreams in a miniskirt",
        "target_zh": "男子穿短裙丝袜街边直播",
        "expand_turns": [0, 1, 7, 8],
        "text_en": {
            0: "Is a man in a skirt breaking the law?",
            1: "The Civil Code.",
            2: "Police do not enforce the Civil Code.",
            3: "Work it out yourself.",
            4: "I am police. Does the Civil Code cover other policing?",
            5: "If you are police, listen instead of being sarcastic.",
            6: "I am discussing, not being sarcastic.",
            7: "He backed off to give you face.",
            8: "I don't need face. This is law. What's the basis? He already quit.",
        },
    },
    2563: {
        "case_id": 2,
        "role": "Speaker history",
        "title": "Case 2 — Speaker history",
        "target_en": "Must delete the ex after a breakup",
        "target_zh": "分手后一定要删除对方",
        "expand_turns": [0, 1, 9],
        "text_en": {
            0: "Why keep the chat? To annoy the new partner?",
            1: "Deleting makes it harder to move on.",
            2: "People who delete still want to check on the ex.",
            3: "Deleting feels like I still cannot let go.",
            4: "Deleting is a clean break.",
            5: "A breakup is already a clean break. I never go back.",
            6: "Then delete it. Why keep the chat?",
            7: "I will delete, but not the second we break up.",
            8: "Saying it first is not winning.",
            9: "I don't see it as winning. We split calmly. I just said why.",
        },
    },
    2201: {
        "case_id": 3,
        "role": "Long-range reply",
        "title": "Case 3 — Long-range reply",
        "target_en": "Opposing Jiang Ge's mother",
        "target_zh": "反对江歌妈妈",
        "expand_turns": [1, 5, 7],
        "text_en": {
            0: "A victim need not stay pitiful forever.",
            1: "Can't the donations support her?",
            2: "Did you not see that she donated?",
            3: "I can send screenshots privately.",
            4: "I meant she donated to others.",
            5: "She donated compensation, not the public funds.",
            6: "Why attack this? Who is the villain here?",
            7: "Who is the villain? Get that straight first.",
        },
    },
    1288: {
        "case_id": 4,
        "role": "Implicit stance",
        "title": "Case 4 — Implicit stance",
        "target_en": "Disliking Wu Yanni",
        "target_zh": "不喜欢吴艳妮",
        "expand_turns": [0, 5, 6],
        "text_en": {
            0: "Just making the field is already great.",
            1: "Then men's football too? Great?",
            2: "Did they make the Olympics?",
            3: "They did in 2008, and scored.",
            4: "This Olympics. Men's football has little to brag about.",
            5: "Don't compare a niche event with football.",
            6: "What? Did I start that comparison?",
        },
    },
}

OFFLINE_TURNS: Dict[int, List[Dict[str, Any]]] = {
    681: [
        {"turn": 0, "speaker_id": 0, "gold_turn_name": "favor", "text": "男的穿丝袜短裙会触犯什么法律吗？"},
        {"turn": 1, "speaker_id": 1, "gold_turn_name": "against", "text": "有一个东西叫民法典"},
        {"turn": 2, "speaker_id": 10, "gold_turn_name": "favor", "text": "可是民法典里面规定的内容不应该是法院来决定的吗，警察怎么会根据民法典规定内容执法呢，不应该是治安管理处罚法或者刑法吗"},
        {"turn": 3, "speaker_id": 1, "gold_turn_name": "neutral", "text": "我不知道说什么好，给我整不会了，哎，还是您自己体会吧"},
        {"turn": 4, "speaker_id": 10, "gold_turn_name": "favor", "text": "我就是公安啊，我真不知道我怎么体会，你要是知道你告诉一下我呗，民法典里面对高空抛物有提到公安要参与，但是最后涉事还是要用治安管理处罚法或者刑法，其余的民法典里面规定的适用于公安执法吗？"},
        {"turn": 5, "speaker_id": 24, "gold_turn_name": "neutral", "text": "您是警察那就被别人普及一下呗，要么非要在这阴阳怪气"},
        {"turn": 6, "speaker_id": 10, "gold_turn_name": "neutral", "text": "我就是在讨论啊，哪里有阴阳怪气……是他让我自己体会啊，倒是你上来就会指责人"},
        {"turn": 7, "speaker_id": 24, "gold_turn_name": "neutral", "text": "他是见你披着警察这套皮不和您争辩，您自己都搬出来了，能不给您个面子嘛！？"},
        {"turn": 8, "speaker_id": 10, "gold_turn_name": "neutral", "text": "我要这面子有什么用，我觉得这是学术这是法律问题，不是面子问题，是他自己在各种杠，大家给他普法他听吗？合理讨论有人参与吗？那你可以说说啊，这里有什么法律依据…且现在他是不回复吗，他是已经不回复所有人了"},
    ],
    2563: [
        {"turn": 0, "speaker_id": 0, "gold_turn_name": "favor", "text": "为什么不删？留着膈应现任吗"},
        {"turn": 1, "speaker_id": 4, "gold_turn_name": "against", "text": "不觉得删了，反而很难走出来吗？没删反而把他给忘了，一干二净了。"},
        {"turn": 2, "speaker_id": 12, "gold_turn_name": "against", "text": "我是觉得会删的人就是知道自己忍不住会去找前任和了解他的近况"},
        {"turn": 3, "speaker_id": 4, "gold_turn_name": "against", "text": "删了就感觉好像是我还离不开他一样。不删，我会忘得快。"},
        {"turn": 4, "speaker_id": 21, "gold_turn_name": "favor", "text": "删了才是离开的一干二净啊"},
        {"turn": 5, "speaker_id": 4, "gold_turn_name": "against", "text": "既然说分手了，就是会一干二净呀，为什么会觉得有了微信就会有还会有复合呢。反正我是分的很干净的，只要一说分手，我绝不会再找复合。"},
        {"turn": 6, "speaker_id": 21, "gold_turn_name": "favor", "text": "那就删了呗，既然不会复合，心里都不占了，还占那个内存干啥"},
        {"turn": 7, "speaker_id": 4, "gold_turn_name": "neutral", "text": "会删，但不会马上说分手就删，因为我分手，不会吵吵闹闹的。因为都是我先说的分手。"},
        {"turn": 8, "speaker_id": 21, "gold_turn_name": "neutral", "text": "可以可以删其实也可以很安静的删，不一定吵吵闹闹，还有，先分手的也不代表是占上风或者先放下那个不能代表什么"},
        {"turn": 9, "speaker_id": 4, "gold_turn_name": "neutral", "text": "没有那么想，先说分手就沾上风。分手我们都是很和平的，分手我都是说清楚我们分手的原因我们存在什么问题。不会因为说什么我先说的分手而感到怎么样这样子。"},
    ],
    2201: [
        {"turn": 0, "speaker_id": 0, "gold_turn_name": "against", "text": "总不能受害者就必须一辈子活在阴影里，一直可怜兮兮的样子吧，不偷不抢，不拿过去的事吸引热度挣钱，挺好的"},
        {"turn": 1, "speaker_id": 3, "gold_turn_name": "favor", "text": "千万的善款不能养活她吗？还要多少？"},
        {"turn": 2, "speaker_id": 4, "gold_turn_name": "against", "text": "？？？？？没看人家捐款？？？？？？？？？？？"},
        {"turn": 3, "speaker_id": 3, "gold_turn_name": "neutral", "text": "想看捐款截图关注我私发给你，给你看小部分你不震撼我都不信。这里评论不让贴图"},
        {"turn": 4, "speaker_id": 4, "gold_turn_name": "against", "text": "还有你的理解能力有误，我回复你的意思是江歌妈妈对外捐款了"},
        {"turn": 5, "speaker_id": 3, "gold_turn_name": "favor", "text": "你的理解能力也差，我说的是她捐给公益组织的69万，是刘鑫给他的赔偿款，善款一分没捐，看懂了吗？"},
        {"turn": 6, "speaker_id": 25, "gold_turn_name": "against", "text": "这你都能喷也是可以的，巴不得苦命人一辈子命苦，终日以泪洗面，坏人逍遥法外就是对的是吧？你管她拿的是什么钱捐的，捐了是真，捐多少还配你管了"},
        {"turn": 7, "speaker_id": 3, "gold_turn_name": "neutral", "text": "谁是坏人搞清楚了没？搞清楚再来说话"},
    ],
    1288: [
        {"turn": 0, "speaker_id": 0, "gold_turn_name": "against", "text": "能站在赛场就很棒了"},
        {"turn": 1, "speaker_id": 14, "gold_turn_name": "favor", "text": "那国足站在赛场上，棒不棒？"},
        {"turn": 2, "speaker_id": 16, "gold_turn_name": "against", "text": "站在奥运会赛场了吗，真逆天"},
        {"turn": 3, "speaker_id": 14, "gold_turn_name": "neutral", "text": "08年就站了啊，还进了个球呢"},
        {"turn": 4, "speaker_id": 16, "gold_turn_name": "against", "text": "说这届奥运会呢你扯什么08年不过也是哈，男足也就那么一两次能扯"},
        {"turn": 5, "speaker_id": 26, "gold_turn_name": "favor", "text": "拿小众赛道跟大众赛道比，你没事吧？"},
        {"turn": 6, "speaker_id": 16, "gold_turn_name": "neutral", "text": "？说什么呢你 是我先开始比较的吗"},
    ],
}

OFFLINE_GATES: Dict[int, Dict[str, Any]] = {
    681: {
        "relation_importance": {"speaker": 0.2912, "dialogue": 0.1658, "temporal": 0.3345, "reply": 0.2084},
        "target_semantic_preference": {"background": 0.2961, "favor": 0.3246, "against": 0.3793},
        "gold_name": "neutral",
        "pred_name": "neutral",
        "reply_analysis": {"prob_immediate_previous": 0.2516, "max_history_prob": 0.2809, "prob_root": 0.0000},
    },
    2563: {
        "relation_importance": {"speaker": 0.3083, "dialogue": 0.1736, "temporal": 0.3002, "reply": 0.2179},
        "target_semantic_preference": {"background": 0.7555, "favor": 0.0535, "against": 0.1910},
        "gold_name": "neutral",
        "pred_name": "neutral",
        "reply_analysis": {"prob_immediate_previous": 0.1427, "max_history_prob": 0.3577, "prob_root": 0.0001},
    },
    2201: {
        "relation_importance": {"speaker": 0.3050, "dialogue": 0.1874, "temporal": 0.2769, "reply": 0.2307},
        "target_semantic_preference": {"background": 0.2341, "favor": 0.6340, "against": 0.1319},
        "gold_name": "neutral",
        "pred_name": "neutral",
        "reply_analysis": {"prob_immediate_previous": 0.0541, "max_history_prob": 0.4170, "prob_root": 0.0001},
    },
    1288: {
        "relation_importance": {"speaker": 0.2957, "dialogue": 0.1603, "temporal": 0.3359, "reply": 0.2081},
        "target_semantic_preference": {"background": 0.1902, "favor": 0.1740, "against": 0.6358},
        "gold_name": "neutral",
        "pred_name": "neutral",
        "reply_analysis": {"prob_immediate_previous": 0.1976, "max_history_prob": 0.2776, "prob_root": 0.0001},
    },
}


def xml(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def wrap_text(text: str, max_chars: int = 48, max_lines: int = 3) -> List[str]:
    raw = " ".join(str(text or "").split())
    if not raw:
        return [""]
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in raw)
    chunks: List[str] = []
    if has_cjk and " " not in raw:
        for i in range(0, len(raw), max_chars):
            chunks.append(raw[i : i + max_chars])
    else:
        cur: List[str] = []
        for w in raw.split(" "):
            trial = " ".join(cur + [w]) if cur else w
            if len(trial) <= max_chars:
                cur.append(w)
            else:
                if cur:
                    chunks.append(" ".join(cur))
                cur = [w]
        if cur:
            chunks.append(" ".join(cur))
    if len(chunks) > max_lines:
        chunks = chunks[:max_lines]
        last = chunks[-1]
        if not last.endswith("…"):
            chunks[-1] = last[: max(1, max_chars - 1)] + "…"
    return chunks


def temporal_edges(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    return [
        {"src_turn": int(turns[i]["turn"]), "dst_turn": int(turns[i + 1]["turn"])}
        for i in range(len(turns) - 1)
    ]


def speaker_edges(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    by_spk: Dict[int, List[int]] = defaultdict(list)
    for t in turns:
        by_spk[int(t["speaker_id"])].append(int(t["turn"]))
    out = []
    for spk, idxs in sorted(by_spk.items()):
        for a, b in zip(idxs, idxs[1:]):
            out.append({"src_turn": a, "dst_turn": b, "speaker_id": int(spk)})
    return out


def reply_edges_from_extracted(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only Soft Reply metadata. Never consecutive-turn fallback."""
    if case.get("in_doc_reply_edges"):
        return list(case["in_doc_reply_edges"])
    turns = case.get("conversation") or case.get("utterances") or []
    if not turns:
        return []
    uid_to_turn = {int(t["utterance_id"]): int(t["turn"]) for t in turns if "utterance_id" in t}
    last_turn = int(turns[-1]["turn"])
    seen: Set[Tuple[int, int]] = set()
    edges: List[Dict[str, Any]] = []

    def add(src_turn: int, dst_turn: int, prob: float, src_uid: int, dst_uid: int) -> None:
        key = (src_turn, dst_turn)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "src_turn": src_turn,
                "dst_turn": dst_turn,
                "src_utterance_id": src_uid,
                "dst_utterance_id": dst_uid,
                "prob": float(prob),
                "source": "soft_reply_P(j|t)",
            }
        )

    for e in case.get("soft_reply_incoming_edges") or []:
        uid = int(e.get("utterance_id", -1))
        if uid not in uid_to_turn:
            continue
        add(uid_to_turn[uid], last_turn, float(e.get("edge_weight") or 0.0), uid, int(turns[-1].get("utterance_id", -1)))
    for e in case.get("soft_reply_top5") or []:
        if e.get("kind") == "ROOT":
            continue
        uid = int(e.get("utterance_id", -1))
        if uid not in uid_to_turn:
            continue
        add(uid_to_turn[uid], last_turn, float(e.get("prob") or 0.0), uid, int(turns[-1].get("utterance_id", -1)))
    return edges


def expanded_turn_set(n: int, speaker0: int, turns: Sequence[Dict[str, Any]], extra: Iterable[int]) -> Set[int]:
    if n <= 6:
        return set(range(n))
    expand = {0, n - 1}
    for t in extra:
        if 0 <= int(t) < n:
            expand.add(int(t))
    for t in turns:
        if int(t["speaker_id"]) == int(speaker0):
            expand.add(int(t["turn"]))
    return expand


def normalize_turns(raw: Sequence[Dict[str, Any]], text_en: Dict[int, str]) -> List[Dict[str, Any]]:
    out = []
    for i, t in enumerate(raw):
        turn = int(t.get("turn", i))
        gold = str(t.get("gold_turn_name") or t.get("gold") or "")
        out.append(
            {
                "turn": turn,
                "utterance_id": int(t["utterance_id"]) if t.get("utterance_id") is not None else turn,
                "node_index": int(t["node_index"]) if t.get("node_index") is not None else -1,
                "speaker_id": int(t["speaker_id"]),
                "gold_turn_name": gold,
                "text": str(t.get("text") or ""),
                "text_en": str(text_en.get(turn) or t.get("text_en") or t.get("text") or ""),
            }
        )
    out.sort(key=lambda r: r["turn"])
    return out


def merge_extracted(case: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = int(case["doc_id"])
    cat = PAPER_CATALOG.get(doc_id, {})
    turns = normalize_turns(case.get("conversation") or [], cat.get("text_en") or {})
    return {
        "case_id": int(cat.get("case_id") or case.get("case_id") or 0),
        "doc_id": doc_id,
        "role": cat.get("role") or case.get("role") or "",
        "title": cat.get("title") or "Case %s" % doc_id,
        "target_en": cat.get("target_en") or case.get("target") or "",
        "target_zh": cat.get("target_zh") or case.get("target") or "",
        "gold_name": case.get("gold_name"),
        "pred_name": case.get("pred_name"),
        "relation_importance": case.get("relation_importance") or {},
        "target_semantic_preference": case.get("target_semantic_preference") or {},
        "reply_analysis": case.get("reply_analysis") or {},
        "utterances": turns,
        "expand_turns": list(cat.get("expand_turns") or []),
        "temporal_edges": case.get("in_doc_temporal_edges") or temporal_edges(turns),
        "speaker_edges": case.get("in_doc_speaker_edges") or speaker_edges(turns),
        "reply_edges": reply_edges_from_extracted(case),
        "dialogue_members": case.get("dialogue_members") or [t["turn"] for t in turns],
        "reply_source": "soft_reply_P(j|t)",
    }


def offline_payload(doc_id: int) -> Dict[str, Any]:
    cat = PAPER_CATALOG[doc_id]
    meta = OFFLINE_GATES[doc_id]
    turns = normalize_turns(OFFLINE_TURNS[doc_id], cat["text_en"])
    return {
        "case_id": cat["case_id"],
        "doc_id": doc_id,
        "role": cat["role"],
        "title": cat["title"],
        "target_en": cat["target_en"],
        "target_zh": cat["target_zh"],
        "gold_name": meta["gold_name"],
        "pred_name": meta["pred_name"],
        "relation_importance": meta["relation_importance"],
        "target_semantic_preference": meta["target_semantic_preference"],
        "reply_analysis": meta["reply_analysis"],
        "utterances": turns,
        "expand_turns": list(cat["expand_turns"]),
        "temporal_edges": temporal_edges(turns),
        "speaker_edges": speaker_edges(turns),
        "reply_edges": [],
        "dialogue_members": [t["turn"] for t in turns],
        "reply_source": "pending_extract",
    }


class Box:
    def __init__(self, turn: int, x: float, y: float, w: float, h: float, expanded: bool):
        self.turn = turn
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.expanded = expanded

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


def stance_short(name: str) -> str:
    n = (name or "").lower()
    if n.startswith("fav"):
        return "Favor"
    if n.startswith("aga"):
        return "Against"
    if n.startswith("neu"):
        return "Neutral"
    return name or ""


def render_case_svg(payload: Dict[str, Any], width: int = 760) -> str:
    turns = payload["utterances"]
    n = len(turns)
    speaker0 = int(turns[0]["speaker_id"]) if turns else 0
    expand = expanded_turn_set(n, speaker0, turns, payload.get("expand_turns") or [])
    last_turn = n - 1

    node_x = 118.0
    exp_w = 430.0
    cmp_w = 168.0
    top = 92.0
    y = top
    boxes: List[Box] = []
    line_h = 15.0
    for t in turns:
        is_exp = int(t["turn"]) in expand
        if is_exp:
            lines = wrap_text(t["text_en"], 46, 3)
            h = 28.0 + line_h * len(lines)
            w = exp_w
        else:
            h = 28.0
            w = cmp_w
        boxes.append(Box(int(t["turn"]), node_x, y, w, h, is_exp))
        y += h + (12.0 if is_exp else 8.0)
    height = y + 36.0

    speakers_order: List[int] = []
    for t in turns:
        sid = int(t["speaker_id"])
        if sid not in speakers_order:
            speakers_order.append(sid)
    spk_x = {sid: 28.0 + 10.0 * i for i, sid in enumerate(speakers_order)}
    box_of = {b.turn: b for b in boxes}

    parts: List[str] = []
    cid = int(payload["case_id"] or payload["doc_id"])
    marker_id = "replyArrow%d" % cid
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
        'role="img" aria-label="%s">'
        % (width, int(math.ceil(height)), xml(payload["title"]))
    )
    parts.append(
        '<defs><marker id="%s" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="%s"/></marker></defs>'
        % (marker_id, COL_REPLY)
    )

    # Header
    rel = payload.get("relation_importance") or {}
    sem = payload.get("target_semantic_preference") or {}
    ra = payload.get("reply_analysis") or {}
    parts.append(
        '<text x="16" y="22" font-size="14" font-weight="600" fill="%s">%s</text>'
        % (COL_TEXT, xml(payload["title"]))
    )
    parts.append(
        '<text x="16" y="40" font-size="11" fill="%s">Target: %s · doc %s · last %s / %s</text>'
        % (
            COL_MUTED,
            xml(payload.get("target_en") or payload.get("target_zh") or ""),
            xml(payload["doc_id"]),
            xml(payload.get("gold_name") or ""),
            xml(payload.get("pred_name") or ""),
        )
    )
    gate = "Spk %.2f   Dlg %.2f   Tmp %.2f   Rpl %.2f" % (
        float(rel.get("speaker") or 0),
        float(rel.get("dialogue") or 0),
        float(rel.get("temporal") or 0),
        float(rel.get("reply") or 0),
    )
    parts.append(
        '<text x="16" y="56" font-size="11" fill="%s">%s</text>' % (COL_MUTED, xml(gate))
    )
    extra = []
    if sem:
        extra.append(
            "sem %s" % xml(max(sem, key=lambda k: float(sem.get(k) or -1)))
        )
    if ra:
        extra.append(
            "P(prev)=%.2f  P(hist)=%.2f"
            % (
                float(ra.get("prob_immediate_previous") or 0),
                float(ra.get("max_history_prob") or 0),
            )
        )
    if extra:
        parts.append(
            '<text x="16" y="72" font-size="11" fill="%s">%s</text>'
            % (COL_MUTED, xml(" · ".join(extra)))
        )

    # Legend
    lx = 470.0
    ly = 18.0
    legend = [
        (COL_DIALOGUE, "solid", "Dialogue"),
        (COL_TEMPORAL, "solid", "Temporal"),
        (COL_SPEAKER, "dashed", "Speaker history"),
        (COL_REPLY, "solid", "Reply P(j|t)"),
    ]
    for i, (col, style, lab) in enumerate(legend):
        x0 = lx
        y0 = ly + i * 16
        dash = ' stroke-dasharray="5 3"' if style == "dashed" else ""
        parts.append(
            '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
            'stroke-width="2" fill="none"%s/>'
            % (x0, y0, x0 + 22, y0, col, dash)
        )
        parts.append(
            '<text x="%.1f" y="%.1f" font-size="10" fill="%s">%s</text>'
            % (x0 + 28, y0 + 3, COL_TEXT, xml(lab))
        )

    # Dialogue bracket covering every utterance, including compact ones.
    if boxes:
        bx0 = boxes[0].y - 4
        bx1 = boxes[-1].bottom + 4
        parts.append(
            '<path d="M108,%.1f H98 V%.1f H108" fill="none" stroke="%s" '
            'stroke-width="2"/>' % (bx0, bx1, COL_DIALOGUE)
        )
        parts.append(
            '<text x="16" y="%.1f" font-size="10" fill="%s">Dialogue</text>'
            % ((bx0 + bx1) / 2.0, COL_DIALOGUE)
        )

    # Speaker history (left, dashed). May skip compact nodes; those nodes stay.
    for e in payload.get("speaker_edges") or []:
        a = box_of.get(int(e["src_turn"]))
        b = box_of.get(int(e["dst_turn"]))
        if a is None or b is None:
            continue
        sid = int(e.get("speaker_id", turns[a.turn]["speaker_id"]))
        x = spk_x.get(sid, 28.0)
        parts.append(
            '<path d="M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="none" '
            'stroke="%s" stroke-width="1.6" stroke-dasharray="6 4"/>'
            % (a.x - 6, a.cy, x, a.cy, x, b.cy, b.x - 6, b.cy, COL_SPEAKER)
        )

    # Temporal: consecutive only, along the spine.
    for e in payload.get("temporal_edges") or []:
        a = box_of.get(int(e["src_turn"]))
        b = box_of.get(int(e["dst_turn"]))
        if a is None or b is None:
            continue
        x = node_x + 14
        parts.append(
            '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
            'stroke-width="1.4"/>' % (x, a.bottom, x, b.y, COL_TEMPORAL)
        )

    # Reply: metadata only. Arrow still hits a compact placeholder.
    reply_edges = list(payload.get("reply_edges") or [])
    reply_edges.sort(key=lambda e: (-float(e.get("prob") or 0), int(e["src_turn"])))
    for i, e in enumerate(reply_edges):
        a = box_of.get(int(e["src_turn"]))
        b = box_of.get(int(e["dst_turn"]))
        if a is None or b is None:
            continue
        span = abs(b.cy - a.cy)
        bulge = 36.0 + min(90.0, 0.22 * span) + 10.0 * (i % 3)
        x1, y1 = a.right, a.cy
        x2, y2 = b.right, b.cy
        c1x, c2x = x1 + bulge, x2 + bulge
        parts.append(
            '<path d="M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="none" '
            'stroke="%s" stroke-width="1.8" marker-end="url(#%s)"/>'
            % (x1, y1, c1x, y1, c2x, y2, x2, y2, COL_REPLY, marker_id)
        )
        mx = (c1x + c2x) / 2.0
        my = (y1 + y2) / 2.0
        parts.append(
            '<text x="%.1f" y="%.1f" font-size="9" fill="%s">P=%.2f</text>'
            % (mx - 6, my - 4, COL_REPLY, float(e.get("prob") or 0))
        )

    # Nodes
    for t, box in zip(turns, boxes):
        is_last = int(t["turn"]) == last_turn
        if box.expanded:
            fill, stroke, sw = COL_EXPAND, (COL_LAST if is_last else COL_EXPAND_STROKE), (2.2 if is_last else 1.2)
        else:
            fill, stroke, sw = COL_COMPACT, COL_COMPACT_STROKE, 1.1
        parts.append(
            '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="6" '
            'fill="%s" stroke="%s" stroke-width="%.1f"/>'
            % (box.x, box.y, box.w, box.h, fill, stroke, sw)
        )
        head = "u%d · S%s" % (int(t["turn"]), int(t["speaker_id"]))
        gold = stance_short(str(t.get("gold_turn_name") or ""))
        if is_last:
            head += " · pred"
        if gold:
            head += " · " + gold
        parts.append(
            '<text x="%.1f" y="%.1f" font-size="10" font-weight="600" fill="%s">%s</text>'
            % (box.x + 8, box.y + 14, COL_TEXT if box.expanded else COL_MUTED, xml(head))
        )
        if box.expanded:
            for li, line in enumerate(wrap_text(t["text_en"], 46, 3)):
                parts.append(
                    '<text x="%.1f" y="%.1f" font-size="11" fill="%s">%s</text>'
                    % (box.x + 8, box.y + 30 + li * line_h, COL_TEXT, xml(line))
                )

    parts.append("</svg>")
    return "\n".join(parts)


def render_html(payloads: Sequence[Dict[str, Any]], svgs: Sequence[str]) -> str:
    blocks = []
    for p, svg in zip(payloads, svgs):
        blocks.append(
            '<section style="margin:0 0 28px 0;">%s</section>' % svg
        )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TMGCN case study</title>"
        "<style>body{font-family:Segoe UI,Helvetica,Arial,sans-serif;"
        "margin:24px;color:#222;background:#fff;} "
        "svg{max-width:760px;height:auto;display:block;}</style>"
        "</head><body>\n"
        + "\n".join(blocks)
        + "\n</body></html>\n"
    )


def write_figures(payloads: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svgs = []
    written = []
    for p in payloads:
        svg = render_case_svg(p)
        svgs.append(svg)
        cid = int(p.get("case_id") or 0)
        name = "case%d_structure.svg" % cid if cid else "doc%s_structure.svg" % p["doc_id"]
        path = out_dir / name
        path.write_text(svg, encoding="utf-8")
        written.append(str(path))
    html_path = out_dir / "case_study_four_panels.html"
    html_path.write_text(render_html(payloads, svgs), encoding="utf-8")
    written.append(str(html_path))
    viz_json = out_dir / "four_cases_viz.json"
    viz_json.write_text(
        json.dumps({"cases": list(payloads)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written.append(str(viz_json))
    return written


def render_extracted_cases(
    cases: Sequence[Dict[str, Any]],
    out_dir: Path,
    checkpoint: str = "",
) -> List[str]:
    payloads = [merge_extracted(c) for c in cases]
    payloads.sort(key=lambda p: int(p.get("case_id") or 0))
    if checkpoint:
        for p in payloads:
            p["checkpoint"] = checkpoint
    return write_figures(payloads, Path(out_dir))


def load_input(path: Path) -> List[Dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(blob, list):
        cases = blob
    elif isinstance(blob, dict) and "cases" in blob:
        cases = blob["cases"]
    elif isinstance(blob, dict) and "conversation" in blob:
        cases = [blob]
    else:
        raise SystemExit("unrecognized JSON: %s" % path)
    payloads = []
    for c in cases:
        if "utterances" in c and "temporal_edges" in c:
            payloads.append(c)
        else:
            payloads.append(merge_extracted(c))
    payloads.sort(key=lambda p: int(p.get("case_id") or p.get("doc_id") or 0))
    return payloads


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="", help="four_cases.json or four_cases_viz.json")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    args = p.parse_args()
    out = Path(args.out)
    if args.input:
        payloads = load_input(Path(args.input))
    else:
        payloads = [offline_payload(doc) for doc in (681, 2563, 2201, 1288)]
        n_reply = sum(len(p["reply_edges"]) for p in payloads)
        if n_reply == 0:
            print(
                "offline catalog has structure only; Reply arrows appear after "
                "extract writes four_cases.json (P(j|t), not adjacency)",
                flush=True,
            )
    written = write_figures(payloads, out)
    for w in written:
        print("wrote", w, flush=True)


if __name__ == "__main__":
    main()
