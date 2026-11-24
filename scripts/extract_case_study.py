#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract TMGCN case-study weights on the ZS-CSD test set (eval only).

Does not train. Does not change model math. Reads tensors the current
forward already returns:

  fusion_gate     [N, 4] last-layer softmax over Speaker / Dialogue /
                  Temporal / Reply (GatedFusion)
  target_alpha    [N, 3] single-head softmax over Background / Favor / Against
  reply_probs     [N, N+1] masked-softmax P(j|t); col 0 is ROOT
  soft_reply_edge_index / soft_reply_edge_weight  thresholded graph

HypergraphConv intra-edge attention is NOT returned by the current
forward; this script reports hyperedge membership from the COO incidence
instead of inventing those coefficients.

    python3.9 scripts/extract_case_study.py config/config_model5_rd_sem.yaml
    python3.9 scripts/extract_case_study.py config/config_model5_rd_sem.yaml \\
        --ckpt result/GraphTracer_5/<ts>/save/best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_graph_3 import _load_augmented_records
from datasets.dataset_graph_4 import target_graph_collate_fn
from main_graph import build_dataset, build_model, load_config, set_seed

REL_NAMES = ("speaker", "dialogue", "temporal", "reply")
SEM_NAMES = ("background", "favor", "against")
LABEL_NAMES = {0: "favor", 1: "against", 2: "neutral", -100: "ignore"}
OUT_DIR = ROOT / "results" / "case_study_analysis"

# Short paper windows are chosen from ALL test docs. Default: first 5
# utterances (4 if the doc has only 4). Official CSD still scores the
# full-doc last sentence; case-study gold/pred here are the prefix-final
# utterance, because every utterance has a stance label.
SHORT_WINDOW = 5
SHORT_MIN = 4
SHORT_MAX = 6
SHORT_ROLES = (
    (1, "cross_speaker", "Cross-speaker thread"),
    (2, "speaker_history", "Speaker history"),
    (3, "long_range_reply", "Long-range reply"),
    (4, "implicit_stance", "Implicit stance"),
)

PROVIDED_TARGET = "Negative Perceptions of Public-Sector Employment"
PROVIDED_KEYS = (
    "Public-sector jobs offer good benefits",
    "Who wouldn't want one",
    "They are stable, especially during COVID",
    "after maternity leave",
    "Private-sector jobs pay around 5,000",
    "public-sector jobs pay over 10,000",
    "take home less than 3,000",
    "体制",
    "编制",
    "公务员",
    "产假",
    "疫情",
    "铁饭碗",
)


def _f(x: Any) -> float:
    return float(x)


def _floats(xs) -> List[float]:
    return [_f(v) for v in xs]


def find_ckpt(explicit: str, model_type: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit("checkpoint not found: %s" % p)
        return p
    root = ROOT / "result" / model_type
    cands = sorted(
        root.glob("*/save/best_model.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise SystemExit(
            "no best_model.pt under result/%s/*/save/. pass --ckpt" % model_type
        )
    return cands[0]


def load_ckpt(model, path: Path, device: torch.device) -> None:
    blob = torch.load(str(path), map_location=device)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("load missing keys:", list(missing)[:12], flush=True)
    if unexpected:
        print("load unexpected keys:", list(unexpected)[:12], flush=True)


def hyperedge_members(coo: torch.Tensor, node: int) -> List[Dict[str, Any]]:
    """COO [2, E]: row0=node, row1=hyperedge id. Membership of hyperedges that contain `node`."""
    if coo is None or coo.numel() == 0:
        return []
    nodes = coo[0].tolist()
    eids = coo[1].tolist()
    by_e: Dict[int, List[int]] = defaultdict(list)
    hit = set()
    for n, e in zip(nodes, eids):
        by_e[int(e)].append(int(n))
        if int(n) == int(node):
            hit.add(int(e))
    out = []
    for e in sorted(hit):
        members = sorted(set(by_e[e]))
        out.append({"hyperedge_id": e, "n_members": len(members), "member_nodes": members})
    return out


def text_of_node(uid: int, rec: Dict[str, Any], gid_to_text: Dict[int, str]) -> str:
    for s, g in zip(rec.get("sentences") or [], rec.get("global_ids") or []):
        if int(g) == int(uid):
            return str(s)
    return str(gid_to_text.get(int(uid), ""))


def conversation_from_rec(rec: Dict[str, Any], uid_to_node: Dict[int, int]) -> List[Dict[str, Any]]:
    sents = [str(s) for s in (rec.get("sentences") or [])]
    gids = [int(g) for g in (rec.get("global_ids") or [])]
    speakers = [int(s) for s in (rec.get("speakers") or [])]
    labs = [int(x) for x in (rec.get("all_labels") or [])]
    while len(speakers) < len(sents):
        speakers.append(speakers[-1] if speakers else 0)
    while len(labs) < len(sents):
        labs.append(-100)
    rows = []
    for i, sent in enumerate(sents):
        gid = gids[i] if i < len(gids) else -1
        rows.append(
            {
                "turn": i,
                "utterance_id": gid,
                "node_index": uid_to_node.get(gid, -1),
                "speaker_id": speakers[i],
                "text": sent,
                "gold_turn": labs[i],
                "gold_turn_name": LABEL_NAMES.get(labs[i], str(labs[i])),
            }
        )
    return rows


def reply_topk(
    node: int,
    reply_probs: Optional[torch.Tensor],
    utterance_ids: List[int],
    gid_to_text: Dict[int, str],
    rec: Dict[str, Any],
    k: int = 5,
) -> List[Dict[str, Any]]:
    if reply_probs is None:
        return []
    row = reply_probs[node]
    n = row.numel()
    kk = min(k, n)
    vals, idx = torch.topk(row, kk)
    out = []
    gids_here = set(int(g) for g in rec.get("global_ids") or [])
    for p, col in zip(vals.tolist(), idx.tolist()):
        col = int(col)
        if col == 0:
            out.append(
                {
                    "rank": len(out) + 1,
                    "kind": "ROOT",
                    "node_index": -1,
                    "utterance_id": -1,
                    "in_this_doc": False,
                    "text": "[ROOT: no antecedent]",
                    "prob": _f(p),
                }
            )
            continue
        j = col - 1
        uid = int(utterance_ids[j]) if 0 <= j < len(utterance_ids) else -1
        out.append(
            {
                "rank": len(out) + 1,
                "kind": "utterance",
                "node_index": j,
                "utterance_id": uid,
                "in_this_doc": uid in gids_here,
                "text": text_of_node(uid, rec, gid_to_text),
                "prob": _f(p),
            }
        )
    return out


def slice_record(rec: Dict[str, Any], end: int) -> Dict[str, Any]:
    """Keep utterances [0, end). Identity / labels / ids are not remapped."""
    out = dict(rec)
    for key in ("sentences", "speakers", "all_labels", "global_ids"):
        out[key] = list(rec.get(key) or [])[:end]
    return out


def window_reply_focus(
    turns: List[Dict[str, Any]],
    reply_probs: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Soft Reply inside the shown window only. Not temporal adjacency."""
    empty = {
        "argmax_src_turn": None,
        "argmax_src_utterance_id": None,
        "argmax_prob": 0.0,
        "prev_turn": None,
        "prev_prob": 0.0,
        "span": 0,
        "long_range": False,
        "root_prob": 0.0,
        "in_window_hist": [],
    }
    if not turns or reply_probs is None:
        return empty
    dst = turns[-1]
    dn = int(dst["node_index"])
    if dn < 0 or dn >= int(reply_probs.size(0)):
        return empty
    row = reply_probs[dn]
    hist: List[Tuple[int, float, int]] = []
    for src in turns[:-1]:
        sn = int(src["node_index"])
        col = sn + 1
        if sn < 0 or col >= int(row.numel()):
            continue
        hist.append((int(src["turn"]), _f(row[col]), int(src["utterance_id"])))
    prev_turn = int(turns[-2]["turn"]) if len(turns) >= 2 else None
    prev_p = 0.0
    for turn, p, _uid in hist:
        if turn == prev_turn:
            prev_p = p
    argmax = max(hist, key=lambda x: x[1]) if hist else None
    span = 0
    long_range = False
    if argmax is not None:
        span = int(dst["turn"]) - int(argmax[0])
        long_range = span >= 2
    return {
        "argmax_src_turn": None if argmax is None else int(argmax[0]),
        "argmax_src_utterance_id": None if argmax is None else int(argmax[2]),
        "argmax_prob": 0.0 if argmax is None else float(argmax[1]),
        "prev_turn": prev_turn,
        "prev_prob": prev_p,
        "span": span,
        "long_range": long_range,
        "root_prob": _f(row[0]) if int(row.numel()) else 0.0,
        "in_window_hist": [
            {"turn": t, "prob": p, "utterance_id": u}
            for t, p, u in sorted(hist, key=lambda x: -x[1])[:5]
        ],
    }


def last_speaker_count(case: Dict[str, Any]) -> int:
    spk = case.get("last_node_speaker_id")
    if spk is None:
        return 0
    return sum(1 for t in case.get("conversation") or [] if int(t["speaker_id"]) == int(spk))


def target_mentioned(text: str, target: str) -> bool:
    t = str(target or "").strip()
    s = str(text or "")
    if t and t in s:
        return True
    if len(t) >= 4:
        for n in range(len(t) - 1):
            piece = t[n : n + 2]
            if piece and piece in s:
                return True
    return False


def score_cross_speaker(c: Dict[str, Any]) -> float:
    if not c.get("correct") or not (SHORT_MIN <= int(c["n_turns"]) <= SHORT_MAX):
        return -1e9
    if int(c["n_speakers_in_doc"]) < 2:
        return -1e9
    n_last = last_speaker_count(c)
    if n_last >= int(c["n_turns"]) - 1:
        return -1e9
    wr = c.get("window_reply") or {}
    if float(wr.get("prev_prob") or 0.0) >= 0.40:
        return -1e9
    rel = c["relation_importance"]
    score = 0.0
    score += 1.3 * int(c["n_speakers_in_doc"])
    score += 0.8 * float(rel["temporal"])
    score -= 1.2 * (n_last / float(c["n_turns"]))
    score -= 1.0 * float(wr.get("prev_prob") or 0.0)
    score -= 0.4 * float(rel["speaker"])
    return score


def score_speaker_history(c: Dict[str, Any]) -> float:
    if not c.get("correct") or not (SHORT_MIN <= int(c["n_turns"]) <= SHORT_MAX):
        return -1e9
    n_last = last_speaker_count(c)
    if n_last < 2:
        return -1e9
    rel = c["relation_importance"]
    score = 2.4 * float(rel["speaker"]) + 0.6 * n_last
    if c.get("relation_argmax") == "speaker":
        score += 1.0
    return score


def score_long_range_reply(c: Dict[str, Any]) -> float:
    if not c.get("correct") or not (SHORT_MIN <= int(c["n_turns"]) <= SHORT_MAX):
        return -1e9
    wr = c.get("window_reply") or {}
    if not wr.get("long_range"):
        return -1e9
    ap = float(wr.get("argmax_prob") or 0.0)
    pp = float(wr.get("prev_prob") or 0.0)
    if ap < 0.12 or pp >= ap:
        return -1e9
    return 3.2 * ap - 2.4 * pp + 0.35 * float(wr.get("span") or 0)


def score_implicit_stance(c: Dict[str, Any]) -> float:
    if not c.get("correct") or not (SHORT_MIN <= int(c["n_turns"]) <= SHORT_MAX):
        return -1e9
    if target_mentioned(c.get("last_text") or "", c.get("target") or ""):
        return -1e9
    if c.get("semantic_argmax") == "background":
        return -1e9
    sem = c.get("target_semantic_preference") or {}
    name = c.get("semantic_argmax") or "against"
    score = 2.2 * float(sem.get(name) or 0.0)
    if c.get("gold_name") == "neutral":
        score += 0.4
    last = str(c.get("last_text") or "")
    score += 0.25 * max(0.0, 1.0 - len(last) / 80.0)
    return score


ROLE_SCORERS = {
    "cross_speaker": score_cross_speaker,
    "speaker_history": score_speaker_history,
    "long_range_reply": score_long_range_reply,
    "implicit_stance": score_implicit_stance,
}


def pick_short_roles(short_cases: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    tops: Dict[str, List[Dict[str, Any]]] = {}
    selected: Dict[str, Dict[str, Any]] = {}
    used = set()
    for _cid, role, _title in SHORT_ROLES:
        fn = ROLE_SCORERS[role]
        ranked = sorted(short_cases, key=fn, reverse=True)
        kept = [c for c in ranked if fn(c) > -1e8]
        tops[role] = kept[:12]
        pick = None
        for c in kept:
            if int(c["doc_id"]) in used:
                continue
            pick = c
            break
        if pick is not None:
            used.add(int(pick["doc_id"]))
            selected[role] = pick
    return selected, tops


def format_short_hit(c: Dict[str, Any], rank: int = 0) -> str:
    rel = c["relation_importance"]
    wr = c.get("window_reply") or {}
    head = (
        "%s doc=%s target=%s window=%s n=%d full=%s nspk=%d gold/pred=%s/%s "
        "spk=%.3f dlg=%.3f tmp=%.3f rpl=%.3f Pprev=%.3f Pmax=%.3f r_src=u%s span=%s"
        % (
            ("#%d" % rank) if rank else "",
            c["doc_id"],
            (c.get("target") or "")[:24],
            c.get("window"),
            c["n_turns"],
            c.get("full_n_turns"),
            c["n_speakers_in_doc"],
            c.get("gold_name"),
            c.get("pred_name"),
            rel["speaker"],
            rel["dialogue"],
            rel["temporal"],
            rel["reply"],
            float(wr.get("prev_prob") or 0.0),
            float(wr.get("argmax_prob") or 0.0),
            wr.get("argmax_src_turn"),
            wr.get("span"),
        )
    )
    lines = [head]
    conv = c.get("conversation") or []
    last_turn = int(conv[-1]["turn"]) if conv else -1
    for t in conv:
        text = str(t.get("text") or "").replace("\n", " ")
        if len(text) > 72:
            text = text[:72] + "…"
        mark = " <CUR>" if int(t["turn"]) == last_turn else ""
        lines.append(
            "  u%d | S%s | %s | %s%s"
            % (t["turn"], t["speaker_id"], t.get("gold_turn_name"), text, mark)
        )
    return "\n".join(lines)


def in_doc_temporal_edges(turns: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    """Consecutive turns inside this document only."""
    return [
        {"src_turn": int(turns[i]["turn"]), "dst_turn": int(turns[i + 1]["turn"])}
        for i in range(len(turns) - 1)
    ]


def in_doc_speaker_edges(turns: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    """Same-speaker history inside this document. May skip other speakers.

    These are Speaker-hyperedge links, not Reply.
    """
    by_spk: Dict[int, List[int]] = defaultdict(list)
    for t in turns:
        by_spk[int(t["speaker_id"])].append(int(t["turn"]))
    edges = []
    for spk, idxs in sorted(by_spk.items()):
        for a, b in zip(idxs, idxs[1:]):
            edges.append({"src_turn": a, "dst_turn": b, "speaker_id": int(spk)})
    return edges


def in_doc_soft_reply_edges(
    turns: List[Dict[str, Any]],
    reply_probs: Optional[torch.Tensor],
    threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """Soft Reply Graph restricted to this document.

    Uses ReplyPredictor ``P(j|t)``. Does not use temporal adjacency.
    Keeps P>threshold, or the in-doc argmax if every history mass is below
    the threshold (same fallback as ``build_soft_reply_graph``).
    ROOT is not a graph node and is not drawn.
    """
    if reply_probs is None:
        return []
    edges: List[Dict[str, Any]] = []
    for dst in turns:
        dn = int(dst["node_index"])
        if dn < 0 or dn >= int(reply_probs.size(0)):
            continue
        row = reply_probs[dn]
        hist: List[Tuple[Dict[str, Any], float]] = []
        for src in turns:
            sn = int(src["node_index"])
            if sn < 0 or int(src["turn"]) >= int(dst["turn"]):
                continue
            col = sn + 1
            if col >= int(row.numel()):
                continue
            hist.append((src, _f(row[col])))
        kept = [(s, p) for s, p in hist if p > threshold]
        if not kept and hist:
            kept = [max(hist, key=lambda x: x[1])]
        for src, p in kept:
            edges.append(
                {
                    "src_turn": int(src["turn"]),
                    "dst_turn": int(dst["turn"]),
                    "src_utterance_id": int(src["utterance_id"]),
                    "dst_utterance_id": int(dst["utterance_id"]),
                    "src_speaker_id": int(src["speaker_id"]),
                    "dst_speaker_id": int(dst["speaker_id"]),
                    "prob": p,
                    "source": "soft_reply_P(j|t)",
                }
            )
    return edges


def incoming_soft_edges(
    node: int,
    edge_index: Optional[torch.Tensor],
    edge_weight: Optional[torch.Tensor],
    utterance_ids: List[int],
    gid_to_text: Dict[int, str],
    rec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if edge_index is None or edge_index.numel() == 0:
        return []
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    w = edge_weight.tolist() if edge_weight is not None else [1.0] * len(src)
    gids_here = set(int(g) for g in rec.get("global_ids") or [])
    rows = []
    for j, t, ww in zip(src, dst, w):
        if int(t) != int(node):
            continue
        uid = int(utterance_ids[int(j)]) if 0 <= int(j) < len(utterance_ids) else -1
        rows.append(
            {
                "from_node": int(j),
                "to_node": int(t),
                "utterance_id": uid,
                "in_this_doc": uid in gids_here,
                "text": text_of_node(uid, rec, gid_to_text),
                "edge_weight": _f(ww),
            }
        )
    rows.sort(key=lambda r: r["edge_weight"], reverse=True)
    return rows


def relation_dict(gate_row) -> Dict[str, float]:
    vals = _floats(gate_row)
    while len(vals) < 4:
        vals.append(0.0)
    return {n: vals[i] for i, n in enumerate(REL_NAMES)}


def semantic_dict(alpha_row) -> Dict[str, float]:
    if alpha_row is None:
        return {n: None for n in SEM_NAMES}  # type: ignore
    vals = _floats(alpha_row)
    while len(vals) < 3:
        vals.append(0.0)
    return {n: vals[i] for i, n in enumerate(SEM_NAMES)}


def argmax_name(d: Dict[str, float]) -> str:
    return max(d.items(), key=lambda kv: kv[1])[0]


def is_readable(rec: Dict[str, Any]) -> bool:
    sents = [str(s).strip() for s in (rec.get("sentences") or [])]
    if len(sents) < 3:
        return False
    last = sents[-1]
    if len(last) < 6:
        return False
    joined = " ".join(sents)
    if len(joined) < 24:
        return False
    return True


def dialogue_score(case: Dict[str, Any]) -> float:
    rel = case["relation_importance"]
    g_spk, g_dlg, g_temp, g_rep = (
        rel["speaker"],
        rel["dialogue"],
        rel["temporal"],
        rel["reply"],
    )
    n_spk = int(case["n_speakers_in_doc"])
    n_turn = int(case["n_turns"])
    gold_eq = 1.0 if case["gold"] == case["pred"] and case["gold"] >= 0 else 0.0
    dlg_is_max = 1.0 if argmax_name(rel) == "dialogue" else 0.0
    speaker_dom = max(0.0, g_spk - g_dlg)
    reply = case.get("reply_analysis") or {}
    max_hist = float(reply.get("max_history_prob") or 0.0)
    prev_p = float(reply.get("prob_immediate_previous") or 0.0)
    # Dialogue case wants many speakers, distributed reply, dialogue gate on top.
    score = 0.0
    score += 3.0 * gold_eq
    score += 4.0 * g_dlg
    score += 5.0 * dlg_is_max
    score += 0.6 * min(n_spk, 5) / 5.0
    score += 0.4 * min(n_turn, 8) / 8.0
    score -= 2.5 * speaker_dom
    score -= 2.0 * max(0.0, g_temp - g_dlg)
    score -= 1.2 * max(0.0, prev_p - 0.25)
    score -= 0.8 * max(0.0, max_hist - 0.40)
    if n_spk < 2:
        score -= 2.0
    if not case.get("readable"):
        score -= 1.5
    if not dlg_is_max:
        score -= 4.0
    _ = g_temp, g_rep
    return score


def match_provided(rec: Dict[str, Any]) -> float:
    tgt = str(rec.get("target") or "")
    blob = tgt + "\n" + "\n".join(str(s) for s in (rec.get("sentences") or []))
    score = 0.0
    if PROVIDED_TARGET.lower() in tgt.lower() or tgt.lower() in PROVIDED_TARGET.lower():
        score += 5.0
    for k in PROVIDED_KEYS:
        if k and k.lower() in blob.lower():
            score += 1.0
    return score


def extract_case(
    rec: Dict[str, Any],
    node: int,
    gold: int,
    pred: int,
    fusion_gate: torch.Tensor,
    target_alpha: Optional[torch.Tensor],
    reply_probs: Optional[torch.Tensor],
    soft_ei: Optional[torch.Tensor],
    soft_ew: Optional[torch.Tensor],
    speaker_coo: torch.Tensor,
    dialogue_coo: torch.Tensor,
    utterance_ids: List[int],
    speaker_ids: List[int],
    gid_to_text: Dict[int, str],
    uid_to_node: Dict[int, int],
    target_text: str,
    target_type: str,
) -> Dict[str, Any]:
    rel = relation_dict(fusion_gate[node].tolist())
    sem = semantic_dict(None if target_alpha is None else target_alpha[node].tolist())
    turns = conversation_from_rec(rec, uid_to_node)
    speakers_in_doc = sorted({int(t["speaker_id"]) for t in turns})
    last_gid = int(rec["global_ids"][-1]) if rec.get("global_ids") else -1
    prev_gid = int(rec["global_ids"][-2]) if len(rec.get("global_ids") or []) >= 2 else -1
    prev_node = uid_to_node.get(prev_gid, -1)

    top5 = reply_topk(node, reply_probs, utterance_ids, gid_to_text, rec, k=5)
    incoming = incoming_soft_edges(node, soft_ei, soft_ew, utterance_ids, gid_to_text, rec)
    root_p = 0.0
    max_hist = 0.0
    prev_p = 0.0
    if reply_probs is not None:
        row = reply_probs[node]
        root_p = _f(row[0])
        hist = row[1:]
        if hist.numel():
            max_hist = _f(hist.max())
        if prev_node >= 0 and prev_node + 1 < row.numel():
            prev_p = _f(row[prev_node + 1])

    reply_dom = False
    if incoming:
        top_w = incoming[0]["edge_weight"]
        reply_dom = top_w >= 0.45 and (prev_p >= 0.35 or incoming[0].get("utterance_id") == prev_gid)

    spk_he = hyperedge_members(speaker_coo, node)
    dlg_he = hyperedge_members(dialogue_coo, node)
    doc_nodes = [t["node_index"] for t in turns if t["node_index"] >= 0]
    dlg_members_in_doc = []
    for he in dlg_he:
        dlg_members_in_doc.append(
            {
                "hyperedge_id": he["hyperedge_id"],
                "n_members_total": he["n_members"],
                "member_nodes_in_this_doc": [n for n in he["member_nodes"] if n in doc_nodes],
                "n_extra_nodes_outside_doc": len([n for n in he["member_nodes"] if n not in doc_nodes]),
            }
        )

    suitable_notes = []
    dlg_max = argmax_name(rel) == "dialogue"
    if not dlg_max:
        suitable_notes.append("dialogue_not_argmax")
    if rel["speaker"] > rel["dialogue"]:
        suitable_notes.append("speaker_dominates_dialogue")
    if prev_p >= 0.35:
        suitable_notes.append("strong_immediate_reply")
    if max_hist >= 0.50:
        suitable_notes.append("one_antecedent_dominates")
    if len(speakers_in_doc) < 2:
        suitable_notes.append("single_speaker")
    if gold != pred:
        suitable_notes.append("pred_mismatch")
    if not is_readable(rec):
        suitable_notes.append("low_readability")

    return {
        "target": target_text,
        "target_type": target_type,
        "doc_id": int(rec["doc_id"]),
        "last_node_index": int(node),
        "last_utterance_id": last_gid,
        "last_text": turns[-1]["text"] if turns else "",
        "gold": int(gold),
        "pred": int(pred),
        "gold_name": LABEL_NAMES.get(int(gold), str(gold)),
        "pred_name": LABEL_NAMES.get(int(pred), str(pred)),
        "correct": int(gold) == int(pred) and int(gold) >= 0,
        "n_turns": len(turns),
        "n_speakers_in_doc": len(speakers_in_doc),
        "speakers_in_doc": speakers_in_doc,
        "readable": is_readable(rec),
        "relation_importance": rel,
        "relation_argmax": argmax_name(rel),
        "target_semantic_preference": sem,
        "semantic_argmax": argmax_name({k: (v if v is not None else -1.0) for k, v in sem.items()}),
        "aggregation": {
            "fusion_gate": (
                "last GNN layer only (num_layers=2); single MLP gate; "
                "softmax over 4 channels; values are the raw softmax, not re-normalized; "
                "fused = 4 * sum g_i m_i"
            ),
            "target_alpha": (
                "single-head scaled-dot-product over 3 views "
                "[Background, Favor, Against]; softmax on views; not multi-head"
            ),
            "soft_reply": (
                "ReplyPredictor masked-softmax P(j|t) over ROOT + in-doc history; "
                "graph keeps P>0.05 (or the max if all fall below)"
            ),
            "hypergraph_conv_attention": (
                "PyG HypergraphConv(use_attention=True) coefficients are NOT "
                "exported; membership below is the COO incidence the conv uses"
            ),
        },
        "speaker_hyperedges": spk_he,
        "dialogue_hyperedges": dlg_he,
        "dialogue_hyperedges_restricted_to_doc": dlg_members_in_doc,
        "soft_reply_incoming_edges": incoming,
        "soft_reply_top5": top5,
        "reply_analysis": {
            "prob_root": root_p,
            "max_history_prob": max_hist,
            "prob_immediate_previous": prev_p,
            "immediate_previous_utterance_id": prev_gid,
            "immediate_previous_node": prev_node,
            "dominated_by_direct_or_one_reply": reply_dom,
        },
        "suitable_dialogue_flags": suitable_notes,
        "suitable_dialogue": (
            dlg_max
            and rel["speaker"] < rel["dialogue"]
            and not reply_dom
            and len(speakers_in_doc) >= 2
            and gold == pred
            and is_readable(rec)
        ),
        "conversation": turns,
        "last_node_speaker_id": int(speaker_ids[node]) if 0 <= node < len(speaker_ids) else None,
        "in_doc_temporal_edges": in_doc_temporal_edges(turns),
        "in_doc_speaker_edges": in_doc_speaker_edges(turns),
        "in_doc_reply_edges": in_doc_soft_reply_edges(turns, reply_probs),
        "dialogue_members": [int(t["turn"]) for t in turns],
        "window_reply": window_reply_focus(turns, reply_probs),
    }


def run_split(
    model, dataset, records_by_target, device, prefix_len: int = SHORT_WINDOW
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    cases: List[Dict[str, Any]] = []
    short_cases: List[Dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in target_graph_collate_fn([item]).items()}
            out = model(**batch)
            logits = out["logits"]
            preds = logits.argmax(dim=-1).cpu()
            fusion_gate = out.get("fusion_gate")
            if fusion_gate is None:
                raise RuntimeError("forward did not return fusion_gate")
            fusion_gate = fusion_gate.detach().cpu()
            alpha = out.get("target_alpha")
            if alpha is not None:
                alpha = alpha.detach().cpu()
            reply_probs = out.get("reply_probs")
            if reply_probs is not None:
                reply_probs = reply_probs.detach().cpu()
            soft_ei = out.get("soft_reply_edge_index")
            if soft_ei is not None:
                soft_ei = soft_ei.detach().cpu()
            soft_ew = out.get("soft_reply_edge_weight")
            if soft_ew is not None:
                soft_ew = soft_ew.detach().cpu()
            speaker_coo = batch["speaker_hyperedge_index"].cpu()
            dialogue_coo = batch["dialogue_hyperedge_index"].cpu()
            utterance_ids = batch["utterance_ids"].cpu().tolist()
            speaker_ids = batch["speaker_ids"].cpu().tolist()
            uid_to_node = {int(u): i for i, u in enumerate(utterance_ids)}
            gid_to_text: Dict[int, str] = {}
            target_text = dataset.target_texts[idx]
            target_type = getattr(dataset, "target_types", [""] * len(dataset))[idx]
            recs = records_by_target.get(target_text, [])
            for rec in recs:
                for s, g in zip(rec.get("sentences") or [], rec.get("global_ids") or []):
                    gid_to_text[int(g)] = str(s)
            rec_by_doc = {int(r["doc_id"]): r for r in recs}
            final_docs = batch["doc_final_doc_ids"].cpu().tolist()
            final_nodes = batch["doc_final_node_index"].cpu().tolist()
            final_true = batch["doc_final_true"].cpu().tolist()
            def _one(rec, node, gold, pred, window, full_n):
                case = extract_case(
                    rec=rec,
                    node=int(node),
                    gold=int(gold),
                    pred=int(pred),
                    fusion_gate=fusion_gate,
                    target_alpha=alpha,
                    reply_probs=reply_probs,
                    soft_ei=soft_ei,
                    soft_ew=soft_ew,
                    speaker_coo=speaker_coo,
                    dialogue_coo=dialogue_coo,
                    utterance_ids=utterance_ids,
                    speaker_ids=speaker_ids,
                    gid_to_text=gid_to_text,
                    uid_to_node=uid_to_node,
                    target_text=target_text,
                    target_type=str(target_type),
                )
                case["window"] = window
                case["full_n_turns"] = int(full_n)
                case["provided_match_score"] = match_provided(rec)
                case["dialogue_score"] = dialogue_score(case)
                return case

            for doc, node, gold in zip(final_docs, final_nodes, final_true):
                rec = rec_by_doc.get(int(doc))
                if rec is None:
                    continue
                node = int(node)
                full_n = len(rec.get("sentences") or [])
                cases.append(
                    _one(rec, node, int(gold), int(preds[node]), "doc_final", full_n)
                )
                if full_n < SHORT_MIN:
                    continue
                L = min(int(prefix_len), full_n)
                if L < SHORT_MIN:
                    continue
                prec = slice_record(rec, L)
                gids = prec.get("global_ids") or []
                labs = [int(x) for x in (prec.get("all_labels") or [])]
                if not gids or not labs or int(labs[-1]) < 0:
                    continue
                pnode = uid_to_node.get(int(gids[-1]))
                if pnode is None:
                    continue
                window = "doc_final" if L == full_n else "prefix"
                short_cases.append(
                    _one(prec, pnode, int(labs[-1]), int(preds[int(pnode)]), window, full_n)
                )
            print(
                "extracted target %d/%d %s docs=%d"
                % (idx + 1, len(dataset), target_text[:40], len(final_docs)),
                flush=True,
            )
    return cases, short_cases


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flatten_case(c: Dict[str, Any], rank: Optional[int] = None) -> Dict[str, Any]:
    rel = c["relation_importance"]
    sem = c["target_semantic_preference"]
    ra = c.get("reply_analysis") or {}
    return {
        "rank": rank if rank is not None else "",
        "doc_id": c["doc_id"],
        "target": c["target"],
        "n_turns": c["n_turns"],
        "n_speakers": c["n_speakers_in_doc"],
        "gold": c["gold_name"],
        "pred": c["pred_name"],
        "correct": c["correct"],
        "speaker": "%.4f" % rel["speaker"],
        "dialogue": "%.4f" % rel["dialogue"],
        "temporal": "%.4f" % rel["temporal"],
        "reply": "%.4f" % rel["reply"],
        "relation_argmax": c["relation_argmax"],
        "background": "" if sem["background"] is None else "%.4f" % sem["background"],
        "favor": "" if sem["favor"] is None else "%.4f" % sem["favor"],
        "against": "" if sem["against"] is None else "%.4f" % sem["against"],
        "semantic_argmax": c["semantic_argmax"],
        "prob_root": "%.4f" % float(ra.get("prob_root") or 0.0),
        "prob_immediate_previous": "%.4f" % float(ra.get("prob_immediate_previous") or 0.0),
        "max_history_prob": "%.4f" % float(ra.get("max_history_prob") or 0.0),
        "suitable_dialogue": c["suitable_dialogue"],
        "flags": "|".join(c.get("suitable_dialogue_flags") or []),
        "dialogue_score": "%.4f" % float(c.get("dialogue_score") or 0.0),
        "last_text": c["last_text"],
    }


def write_md(
    path: Path,
    ckpt: Path,
    provided: Optional[Dict[str, Any]],
    top10: List[Dict[str, Any]],
    recommended: Optional[Dict[str, Any]],
) -> None:
    lines = []
    lines.append("# Case Study Weight Extraction")
    lines.append("")
    lines.append("Eval-only. No retraining. Weights are the tensors the current forward already returns.")
    lines.append("")
    lines.append("## Checkpoint")
    lines.append("")
    lines.append("- `%s`" % ckpt.as_posix())
    lines.append("")
    lines.append("## Where the numbers come from")
    lines.append("")
    lines.append("| Quantity | Tensor | Shape | Aggregation |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| Relation importance | `fusion_gate` | `[N, 4]` | last GNN layer; MLP gate; "
        "softmax over `[Speaker, Dialogue, Temporal, Reply]`; **raw softmax, not re-normalized** |"
    )
    lines.append(
        "| Target semantics | `target_alpha` | `[N, 3]` | `TargetGrounding` single-head "
        "scaled-dot-product; softmax over `[Background, Favor, Against]` |"
    )
    lines.append(
        "| Soft reply | `reply_probs` / `soft_reply_edge_weight` | `[N, N+1]` / `[E]` | "
        "masked softmax `P(j\\|t)`; graph keeps `P>0.05` |"
    )
    lines.append(
        "| Speaker / Dialogue membership | COO `*_hyperedge_index` | `[2, E]` | "
        "incidence, not HypergraphConv attention (those coefficients are **not exported**) |"
    )
    lines.append("")
    lines.append("Weights are always taken at the **official last-sentence node** of the document (`doc_final_node_index`), never a conversation average.")
    lines.append("")
    lines.append("### Topology (code, not names)")
    lines.append("")
    lines.append("- **Speaker hyperedge**: all unique-sentence nodes of one speaker **inside the whole target graph** (not just this thread).")
    lines.append("- **Dialogue hyperedge**: one hyperedge per `doc_id`; members are that conversation's utterances. A shared sentence belongs to every doc that contains it.")
    lines.append("- **Temporal**: consecutive turns inside each conversation.")
    lines.append("- **Reply**: Student `P(j|t)` on the original in-conversation prefix candidates.")
    lines.append("")
    lines.append("## Provided Dialogue candidate")
    lines.append("")
    if provided is None:
        lines.append("No test document matched the English target / key sentences. ZS-CSD utterances are Chinese; the provided lines look like a translation. Closest keyword hits were still scored in the JSON.")
    else:
        rel = provided["relation_importance"]
        ra = provided["reply_analysis"]
        lines.append("- target: `%s`" % provided["target"])
        lines.append("- doc_id: `%s`" % provided["doc_id"])
        lines.append("- gold / pred: **%s / %s**" % (provided["gold_name"], provided["pred_name"]))
        lines.append("- speakers in thread: %d, turns: %d" % (provided["n_speakers_in_doc"], provided["n_turns"]))
        lines.append(
            "- relations: Speaker=%.4f Dialogue=%.4f Temporal=%.4f Reply=%.4f (argmax=%s)"
            % (rel["speaker"], rel["dialogue"], rel["temporal"], rel["reply"], provided["relation_argmax"])
        )
        lines.append(
            "- last-utterance reply: P(ROOT)=%.4f P(prev turn)=%.4f max history=%.4f"
            % (ra["prob_root"], ra["prob_immediate_previous"], ra["max_history_prob"])
        )
        lines.append("- suitable_dialogue: `%s` flags=`%s`" % (provided["suitable_dialogue"], ",".join(provided["suitable_dialogue_flags"])))
        if provided["relation_argmax"] == "reply" or ra["prob_immediate_previous"] >= 0.35:
            lines.append("- **Verdict: not a clean Dialogue case** — last utterance leans on Reply / the previous turn, not Dialogue-level context.")
        elif provided["relation_argmax"] == "dialogue" and provided["suitable_dialogue"]:
            lines.append("- **Verdict: usable as Dialogue case.**")
        else:
            lines.append("- **Verdict: weak Dialogue case** — see flags.")
    lines.append("")
    lines.append("## Top-10 Dialogue-context candidates")
    lines.append("")
    if not top10:
        lines.append("None.")
    else:
        lines.append("| rank | doc_id | speakers | turns | gold/pred | Spk | Dlg | Tmp | Rpl | last utterance |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, c in enumerate(top10, 1):
            rel = c["relation_importance"]
            last = (c["last_text"] or "").replace("|", "/")
            if len(last) > 48:
                last = last[:48] + "…"
            lines.append(
                "| %d | %s | %d | %d | %s/%s | %.3f | %.3f | %.3f | %.3f | %s |"
                % (
                    i,
                    c["doc_id"],
                    c["n_speakers_in_doc"],
                    c["n_turns"],
                    c["gold_name"],
                    c["pred_name"],
                    rel["speaker"],
                    rel["dialogue"],
                    rel["temporal"],
                    rel["reply"],
                    last,
                )
            )
    lines.append("")
    lines.append("## Recommended Case 4")
    lines.append("")
    if recommended is None:
        lines.append("No candidate passed the Dialogue filter.")
    else:
        lines.append("Use **doc_id=%s** / target=`%s`." % (recommended["doc_id"], recommended["target"]))
        rel = recommended["relation_importance"]
        sem = recommended["target_semantic_preference"]
        lines.append("")
        lines.append("Relation importance (last utterance, raw softmax):")
        lines.append("")
        lines.append("| Speaker | Dialogue | Temporal | Reply |")
        lines.append("|---|---|---|---|")
        lines.append("| %.4f | %.4f | %.4f | %.4f |" % (rel["speaker"], rel["dialogue"], rel["temporal"], rel["reply"]))
        lines.append("")
        lines.append("Target semantic preference (last utterance, raw softmax):")
        lines.append("")
        lines.append("| Background | Favor | Against |")
        lines.append("|---|---|---|")
        lines.append("| %.4f | %.4f | %.4f |" % (sem["background"], sem["favor"], sem["against"]))
        lines.append("")
        lines.append("Cases 1–3 (Speaker History / Long-range Reply / Implicit Stance) are **not selected in this run**. Their rows in `case_study_weights.csv` are left empty on purpose.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default=str(ROOT / "config" / "config_model5_rd_sem.yaml"))
    p.add_argument("--ckpt", default="")
    p.add_argument("--cuda-index", type=int, default=None)
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--prefix-len", type=int, default=SHORT_WINDOW)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.cuda_index is not None:
        cfg.cuda_index = args.cuda_index
    set_seed(cfg.seed)
    device = torch.device("cuda:%d" % cfg.cuda_index if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.cuda_index)

    ckpt = find_ckpt(args.ckpt, cfg.model_type)
    print("ckpt=%s device=%s" % (ckpt, device), flush=True)

    test_ds = build_dataset(cfg, cfg.test_path)
    cfg.input_size = test_ds.embedding_store.dim
    records = _load_augmented_records(cfg.test_path)
    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_target[str(rec.get("target") or "")].append(rec)

    model = build_model(cfg)
    load_ckpt(model, ckpt, device)
    model.to(device)
    model.eval()

    cases, short_cases = run_split(
        model, test_ds, by_target, device, prefix_len=int(args.prefix_len)
    )
    if not cases:
        raise SystemExit("no last-sentence docs extracted")

    provided_hits = sorted(cases, key=lambda c: c["provided_match_score"], reverse=True)
    provided = provided_hits[0] if provided_hits and provided_hits[0]["provided_match_score"] >= 2.0 else None

    ranked = sorted(cases, key=lambda c: c["dialogue_score"], reverse=True)
    preferred = [
        c
        for c in ranked
        if c["correct"]
        and c["relation_argmax"] == "dialogue"
        and c["n_speakers_in_doc"] >= 2
        and c["n_turns"] >= 3
        and c["readable"]
        and not (c.get("reply_analysis") or {}).get("dominated_by_direct_or_one_reply")
    ]
    if not preferred:
        preferred = [
            c
            for c in ranked
            if c["relation_argmax"] == "dialogue" and c["n_speakers_in_doc"] >= 2
        ]
    pool = preferred if preferred else ranked
    top10 = pool[:10]
    recommended = None
    for c in top10:
        if c["suitable_dialogue"]:
            recommended = c
            break
    if recommended is None and top10:
        recommended = top10[0]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "all_last_sentences.csv",
        [flatten_case(c) for c in cases],
        fieldnames=list(flatten_case(cases[0]).keys()),
    )
    (out / "case4_candidates.json").write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt),
                "n_docs": len(cases),
                "provided_target_query": PROVIDED_TARGET,
                "candidates": top10,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(
        out / "case4_candidates.csv",
        [flatten_case(c, i + 1) for i, c in enumerate(top10)],
        fieldnames=list(flatten_case(top10[0], 1).keys()) if top10 else ["rank"],
    )
    selected = {
        "provided_query": {
            "target": PROVIDED_TARGET,
            "key_sentences": list(PROVIDED_KEYS[:6]),
            "matched": provided is not None,
        },
        "provided_case": provided,
        "recommended_case": recommended,
        "checkpoint": str(ckpt),
        "weight_source": {
            "fusion_gate": "models/model_graph_5_2.py GatedFusion softmax, last layer",
            "target_alpha": "models/model_graph_5_2.py TargetGrounding, views B/F/A",
            "soft_reply": "modules/reply_distillation.py ReplyPredictor + build_soft_reply_graph",
            "hyperedges": "datasets/dataset_graph_4.py speaker/dialogue COO incidence",
        },
    }
    (out / "case4_selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    weight_rows = []
    for cid, role in (
        (1, "Speaker History — not selected this run"),
        (2, "Long-range Reply — not selected this run"),
        (3, "Implicit Stance — not selected this run"),
        (4, "Dialogue Context"),
    ):
        row = {
            "case_id": cid,
            "role": role,
            "status": "empty",
            "doc_id": "",
            "target": "",
            "gold": "",
            "pred": "",
            "speaker": "",
            "dialogue": "",
            "temporal": "",
            "reply": "",
            "background": "",
            "favor": "",
            "against": "",
            "notes": "",
        }
        if cid == 4 and recommended is not None:
            rel = recommended["relation_importance"]
            sem = recommended["target_semantic_preference"]
            row.update(
                {
                    "status": "recommended",
                    "doc_id": recommended["doc_id"],
                    "target": recommended["target"],
                    "gold": recommended["gold_name"],
                    "pred": recommended["pred_name"],
                    "speaker": "%.4f" % rel["speaker"],
                    "dialogue": "%.4f" % rel["dialogue"],
                    "temporal": "%.4f" % rel["temporal"],
                    "reply": "%.4f" % rel["reply"],
                    "background": "%.4f" % sem["background"],
                    "favor": "%.4f" % sem["favor"],
                    "against": "%.4f" % sem["against"],
                    "notes": "argmax=%s; flags=%s"
                    % (recommended["relation_argmax"], "|".join(recommended["suitable_dialogue_flags"])),
                }
            )
        weight_rows.append(row)
        if cid == 4 and provided is not None:
            rel = provided["relation_importance"]
            sem = provided["target_semantic_preference"]
            weight_rows.append(
                {
                    "case_id": "4_provided",
                    "role": "Dialogue Context (user-provided)",
                    "status": "matched" if provided["provided_match_score"] >= 2 else "weak_match",
                    "doc_id": provided["doc_id"],
                    "target": provided["target"],
                    "gold": provided["gold_name"],
                    "pred": provided["pred_name"],
                    "speaker": "%.4f" % rel["speaker"],
                    "dialogue": "%.4f" % rel["dialogue"],
                    "temporal": "%.4f" % rel["temporal"],
                    "reply": "%.4f" % rel["reply"],
                    "background": "%.4f" % sem["background"],
                    "favor": "%.4f" % sem["favor"],
                    "against": "%.4f" % sem["against"],
                    "notes": "argmax=%s; flags=%s; match=%.1f"
                    % (
                        provided["relation_argmax"],
                        "|".join(provided["suitable_dialogue_flags"]),
                        provided["provided_match_score"],
                    ),
                }
            )
    write_csv(
        out / "case_study_weights.csv",
        weight_rows,
        fieldnames=[
            "case_id",
            "role",
            "status",
            "doc_id",
            "target",
            "gold",
            "pred",
            "speaker",
            "dialogue",
            "temporal",
            "reply",
            "background",
            "favor",
            "against",
            "notes",
        ],
    )
    write_md(out / "case_study_analysis.md", ckpt, provided, top10, recommended)

    selected_short, short_tops = pick_short_roles(short_cases)
    short_dump = {
        "prefix_len": int(args.prefix_len),
        "n_short_windows": len(short_cases),
        "reply_source": "in-window Soft Reply P(j|t); not consecutive-turn adjacency",
        "eval_anchor": "prefix-final utterance gold/pred (not necessarily official doc-final)",
        "selected": {},
        "top": {},
    }
    paper_cases = []
    print("==== SHORT TEST SEARCH window=%d (all %d prefix windows) ====" % (int(args.prefix_len), len(short_cases)), flush=True)
    for cid, role, title in SHORT_ROLES:
        hits = short_tops.get(role) or []
        pick = selected_short.get(role)
        short_dump["top"][role] = hits[:8]
        print("\n-- %s (%s)  hits=%d --" % (role, title, len(hits)), flush=True)
        if pick is None:
            print("  no candidate", flush=True)
            continue
        pick = dict(pick)
        pick["case_id"] = cid
        pick["role"] = title
        short_dump["selected"][role] = pick
        paper_cases.append(pick)
        print("PICKED", format_short_hit(pick), flush=True)
        for i, c in enumerate(hits[:5], 1):
            if int(c["doc_id"]) == int(pick["doc_id"]):
                continue
            print(format_short_hit(c, i), flush=True)
    (out / "short_case_candidates.json").write_text(
        json.dumps(short_dump, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if paper_cases:
        (out / "four_cases.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(ckpt),
                    "reply_source": (
                        "in-doc Soft Reply P(j|t)>0.05 from ReplyPredictor; "
                        "not consecutive-turn adjacency"
                    ),
                    "cases": paper_cases,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from visualize_case_study import render_extracted_cases

        viz_paths = render_extracted_cases(paper_cases, out, checkpoint=str(ckpt))
        print("wrote case-study figures: %s" % viz_paths, flush=True)

    print("wrote %s (%d docs, top10=%d)" % (out, len(cases), len(top10)), flush=True)
    if provided is None:
        print("provided English case: NO MATCH on the Chinese test JSON", flush=True)
    else:
        print(
            "provided case doc_id=%s argmax=%s suitable=%s"
            % (provided["doc_id"], provided["relation_argmax"], provided["suitable_dialogue"]),
            flush=True,
        )
    if recommended is not None:
        print(
            "recommend doc_id=%s target=%s dlg=%.4f"
            % (
                recommended["doc_id"],
                recommended["target"][:40],
                recommended["relation_importance"]["dialogue"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
