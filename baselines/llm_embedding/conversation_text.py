#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 CSD JSON 记录还原成可复现的完整会话文本。

每条 JSON 记录 = 一段对话（doc）：
  sentences[i]  第 i 句原文
  speakers[i]   说话人（与句子对齐）
  target        立场对象
  all_labels[i] 第 i 句立场（0 favor / 1 against / 2 neutral / <0 忽略）
  label         文档级标签；官方 CSD 评价用「最后一句」

不编造字段；缺失说话人时写成 speaker_UNK。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


INSTRUCTION = (
    "Represent the stance expressed by the final utterance toward the target, "
    "considering the entire conversation context, speaker identity, and utterance order."
)


def speaker_name(raw: Any) -> str:
    if raw is None:
        return "speaker_UNK"
    if isinstance(raw, int) or (isinstance(raw, str) and str(raw).isdigit()):
        return "speaker_%s" % raw
    text = str(raw).strip()
    return text if text else "speaker_UNK"


def last_sentence_label(record: Dict[str, Any]) -> int:
    """官方 CSD 评价锚点：该会话最后一句的立场。

    与 dataset_graph_3.build_target_graph 一致：优先 doc-level `label`，
    否则用 `all_labels[-1]`。不把该标签复制到其它话语。
    """
    sentences = record.get("sentences") or []
    if not sentences:
        return -100
    doc_label = int(record.get("label", -1))
    all_labels = list(record.get("all_labels") or [])
    last_turn = len(sentences) - 1
    last_occ = int(all_labels[last_turn]) if last_turn < len(all_labels) else doc_label
    return doc_label if doc_label >= 0 else last_occ


def format_conversation(
    record: Dict[str, Any],
    upto_turn: int | None = None,
) -> str:
    """还原完整会话（或前缀 u_0..u_t），保留说话人和原始顺序。"""
    target = str(record.get("target", "") or "").strip() or "(none)"
    sentences: Sequence[str] = record.get("sentences") or []
    speakers: Sequence[Any] = record.get("speakers") or []
    n = len(sentences)
    if n == 0:
        raise ValueError("record has empty sentences: id=%r" % record.get("id", record.get("doc_id")))
    end = n if upto_turn is None else min(int(upto_turn) + 1, n)
    if end <= 0:
        raise ValueError("upto_turn must be >= 0")
    lines = []
    for i in range(end):
        spk = speaker_name(speakers[i] if i < len(speakers) else None)
        marker = " [CURRENT]" if i == end - 1 else ""
        lines.append("[%d] %s:%s %s" % (i + 1, spk, marker, str(sentences[i]).strip()))
    final = str(sentences[end - 1]).strip()
    return (
        "%s\n"
        "\n"
        "Target:\n"
        "%s\n"
        "\n"
        "Conversation:\n"
        "%s\n"
        "\n"
        "Final utterance:\n"
        "%s"
        % (INSTRUCTION, target, "\n".join(lines), final)
    )


def iter_units(record: Dict[str, Any], unit: str) -> List[Dict[str, Any]]:
    """把一条 JSON 记录展开成 baseline 的分类单元。

    last_sentence: 1 个会话向量 ↔ 1 个最后一句标签（官方指标）。
    prefix:        每个 turn 一个向量（编码 u_0..u_t），标签是该 turn 的
                   all_labels[t]，用于话语级、且每条话语表示不同。
    """
    sentences = list(record.get("sentences") or [])
    if not sentences:
        return []
    all_labels = list(record.get("all_labels") or [])
    conv_id = int(record.get("id", record.get("doc_id", -1)))
    target = record.get("target", "")
    target_type = str(record.get("target_type", "") or "")
    if unit == "last_sentence":
        return [
            {
                "conversation_id": conv_id,
                "source_index": None,
                "turn_index": len(sentences) - 1,
                "label": last_sentence_label(record),
                "target": target,
                "target_type": target_type,
                "n_turns": len(sentences),
                "text": format_conversation(record, upto_turn=None),
            }
        ]
    if unit == "prefix":
        rows = []
        for t in range(len(sentences)):
            lab = int(all_labels[t]) if t < len(all_labels) else int(record.get("label", -1))
            rows.append(
                {
                    "conversation_id": conv_id,
                    "source_index": None,
                    "turn_index": t,
                    "label": lab,
                    "target": target,
                    "target_type": target_type,
                    "n_turns": len(sentences),
                    "text": format_conversation(record, upto_turn=t),
                }
            )
        return rows
    raise ValueError("unknown unit=%r (use last_sentence or prefix)" % unit)
