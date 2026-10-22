#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline Stance Teacher: Qwen3-Instruct (or compatible) -> Favor/Against/Neutral.

For each conversation, the Teacher sees the TARGET and a causal prefix
(u_0..u_t, no future turns) and outputs a soft stance distribution for the
CURRENT utterance. Files land in:

    data/stance_teacher/conversation_<id>.json

This is NOT reply-structure Teacher and does NOT run at TMGCN train time.
Generate first, then inspect quality with eval_stance_teacher.py, then train
with use_stance_kd=1.

Examples
--------
# pipeline check (no LLM)
python scripts/generate_stance_teacher.py \\
  --data_path /path/to/train_data.json \\
  --output_dir data/stance_teacher \\
  --provider mock --scope last_sentence

# local Qwen3-8B (Instruct; disable thinking)
python scripts/generate_stance_teacher.py \\
  --data_path /path/to/train_data.json \\
  --output_dir data/stance_teacher \\
  --provider local --model /path/to/Qwen3-8B \\
  --method token --scope last_sentence --device cuda:0

# vLLM / OpenAI-compatible API
python scripts/generate_stance_teacher.py \\
  --data_path /path/to/train_data.json \\
  --output_dir data/stance_teacher \\
  --provider openai --model Qwen3-8B \\
  --api_base http://127.0.0.1:8000/v1 --method json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.llm_embedding.conversation_text import last_sentence_label, speaker_name
from modules.stance_distillation import (
    mock_teacher_probs,
    normalize_stance_probs,
    teacher_argmax,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("generate_stance_teacher")

SYSTEM_PROMPT = (
    "You are an expert at conversational stance detection. "
    "Given a TARGET and a conversation prefix, estimate the CURRENT "
    "utterance's stance toward the TARGET as soft probabilities over "
    "Favor, Against, and Neutral. You never invent future turns."
)

TOKEN_CLASS_NAMES = ("Favor", "Against", "Neutral")


def build_user_prompt(target: str, sentences: Sequence[str], speakers: Sequence[Any], t: int) -> str:
    lines = [
        "Estimate the CURRENT utterance's stance toward the TARGET.",
        "",
        "Target:",
        str(target).strip() or "(none)",
        "",
        "Conversation history visible to you (DO NOT use any later utterance):",
        "",
    ]
    for i in range(t + 1):
        marker = "  [CURRENT]" if i == t else ""
        lines.append("u%d %s:%s" % (i, speaker_name(speakers[i] if i < len(speakers) else None), marker))
        lines.append(str(sentences[i]).strip())
        lines.append("")
    lines.extend(
        [
            "CURRENT utterance is u%d." % t,
            "",
            "Labels (CSD): Favor=0, Against=1, Neutral=2.",
            "Hard constraints:",
            "1. Score only the CURRENT utterance toward the TARGET.",
            "2. You MUST NOT use future utterances (they are not shown).",
            "3. Output SOFT scores (all values >= 0, will be normalized to 1).",
            "",
            'Return JSON only, no markdown: {"favor": <float>, "against": <float>, "neutral": <float>}',
        ]
    )
    return "\n".join(lines)


def build_token_user_prompt(target: str, sentences: Sequence[str], speakers: Sequence[Any], t: int) -> str:
    body = build_user_prompt(target, sentences, speakers, t)
    return (
        body
        + "\n\nNow answer with exactly one word, chosen from Favor, Against, Neutral."
        "\nThe stance of the CURRENT utterance toward the TARGET is"
    )


def _load_records(raw_path: str) -> List[Dict[str, Any]]:
    from datasets.dataset_graph_3 import _load_augmented_records

    return _load_augmented_records(raw_path)


def _extract_json_obj(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _openai_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 180,
    temperature: float = 0.2,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("LLM HTTP %s: %s" % (exc.code, detail)) from exc
    try:
        return body["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError("unexpected LLM response: %s" % body) from exc


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class LocalQwenTeacher:
    """Local HuggingFace causal LM (Qwen3-8B / Instruct-class)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bf16",
        allow_download: bool = False,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        local_only = not allow_download
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_only, trust_remote_code=True
        )
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "auto" else None,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def infer_json(self, target: str, sentences: Sequence[str], speakers: Sequence[Any], t: int) -> List[float]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(target, sentences, speakers, t)},
        ]
        prefix = _apply_chat_template(self.tokenizer, messages)
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].size(1) :], skip_special_tokens=True)
        obj = _extract_json_obj(text)
        return list(normalize_stance_probs(obj))

    def infer_token(self, target: str, sentences: Sequence[str], speakers: Sequence[Any], t: int) -> List[float]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_token_user_prompt(target, sentences, speakers, t)},
        ]
        prefix = _apply_chat_template(self.tokenizer, messages)
        prefix_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        prefix_ids = {k: v.to(self.device) for k, v in prefix_ids.items()}
        logps = []
        for name in TOKEN_CLASS_NAMES:
            class_ids = self.tokenizer(name, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
                self.device
            )
            input_ids = self.torch.cat([prefix_ids["input_ids"], class_ids], dim=1)
            attn = self.torch.ones_like(input_ids)
            labels = input_ids.clone()
            labels[:, : prefix_ids["input_ids"].size(1)] = -100
            with self.torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
            n_tok = int(class_ids.size(1))
            logps.append(-float(out.loss.detach().cpu()) * n_tok)
        scores = self.torch.tensor(logps, dtype=self.torch.float32)
        probs = self.torch.softmax(scores, dim=0)
        return [float(v) for v in probs.tolist()]


class StanceTeacher:
    def __init__(
        self,
        provider: str,
        model: str = "",
        method: str = "token",
        api_base: str = "",
        api_key: str = "",
        device: str = "cuda:0",
        dtype: str = "bf16",
        allow_download: bool = False,
        temperature: float = 0.2,
        max_retries: int = 3,
        sleep_s: float = 0.2,
    ):
        self.provider = str(provider).lower()
        self.model = model
        self.method = str(method).lower()
        self.api_base = api_base or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.sleep_s = float(sleep_s)
        self.local: Optional[LocalQwenTeacher] = None
        if self.provider in ("local", "hf", "huggingface"):
            if not self.model:
                raise ValueError("provider=local 时必须指定 --model（Qwen3-8B 本地目录或 HF id）")
            self.local = LocalQwenTeacher(
                model_path=self.model,
                device=device,
                dtype=dtype,
                allow_download=allow_download,
                temperature=self.temperature,
            )
        elif self.provider in ("openai", "qwen", "vllm") and not self.model:
            raise ValueError("provider=%s 时必须指定 --model" % self.provider)

    def infer(
        self,
        target: str,
        sentences: Sequence[str],
        speakers: Sequence[Any],
        t: int,
        gold: int = -1,
    ) -> List[float]:
        if self.provider in ("mock", "heuristic", "none"):
            return list(mock_teacher_probs(gold))
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.local is not None:
                    if self.method == "json":
                        return self.local.infer_json(target, sentences, speakers, t)
                    return self.local.infer_token(target, sentences, speakers, t)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(target, sentences, speakers, t)},
                ]
                text = _openai_chat(
                    self.api_base, self.api_key, self.model, messages, temperature=self.temperature
                )
                obj = _extract_json_obj(text)
                return list(normalize_stance_probs(obj))
            except Exception as exc:
                last_err = exc
                logger.warning("Teacher failed attempt=%d/%d: %s", attempt, self.max_retries, exc)
                time.sleep(self.sleep_s * attempt)
        logger.error("Teacher retries exhausted, fallback uniform: %s", last_err)
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]


def conversation_out_path(output_dir: Path, conversation_id: Any) -> Path:
    return output_dir / ("conversation_%s.json" % conversation_id)


def gold_at_turn(rec: Dict[str, Any], t: int, n: int) -> int:
    all_labels = list(rec.get("all_labels") or [])
    if t == n - 1:
        return int(last_sentence_label(rec))
    if t < len(all_labels):
        return int(all_labels[t])
    return int(rec.get("label", -1))


def generate_for_conversation(
    rec: Dict[str, Any],
    teacher: StanceTeacher,
    skip_existing: bool,
    output_dir: Path,
    scope: str,
) -> Optional[Dict[str, Any]]:
    cid = int(rec.get("doc_id", rec.get("id", -1)))
    out_path = conversation_out_path(output_dir, cid)
    if skip_existing and out_path.is_file():
        return None
    sentences = list(rec.get("sentences") or [])
    speakers = list(rec.get("speakers") or [])
    global_ids = list(rec.get("global_ids") or [])
    n = len(sentences)
    if n == 0:
        return None
    turns = [n - 1] if str(scope).lower() not in ("all", "all_turns", "prefix") else list(range(n))
    rows: List[Dict[str, Any]] = []
    for t in turns:
        gold = gold_at_turn(rec, t, n)
        probs = teacher.infer(rec.get("target", ""), sentences, speakers, t, gold=gold)
        rows.append(
            {
                "conversation_id": cid,
                "utterance_id": t,
                "global_id": int(global_ids[t]) if t < len(global_ids) else -1,
                "target": rec.get("target", ""),
                "target_type": rec.get("target_type", ""),
                "is_last": t == n - 1,
                "gold_label": gold,
                "probs": {
                    "favor": float(probs[0]),
                    "against": float(probs[1]),
                    "neutral": float(probs[2]),
                },
                "pred": teacher_argmax(probs),
            }
        )
    payload = {
        "conversation_id": cid,
        "target": rec.get("target", ""),
        "target_type": rec.get("target_type", ""),
        "n_turns": n,
        "scope": scope,
        "utterances": rows,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return payload


def _per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 3) -> List[float]:
    scores = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(0.0 if (prec + rec) == 0.0 else 2.0 * prec * rec / (prec + rec))
    return scores


def evaluate_teacher_dir(teacher_dir: Path) -> Dict[str, Any]:
    """Official CSD last-sentence mixed / c / n Macro F1 of Teacher argmax."""
    y_true: List[int] = []
    y_pred: List[int] = []
    types: List[str] = []
    files = sorted(teacher_dir.glob("conversation_*.json"))
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        target_type = str(payload.get("target_type", "") or "")
        for row in payload.get("utterances", []):
            if not row.get("is_last", False):
                continue
            gold = int(row.get("gold_label", -1))
            if gold < 0:
                continue
            y_true.append(gold)
            probs = row.get("probs") or {}
            if "pred" in row:
                pred = int(row["pred"])
            else:
                pred = teacher_argmax(
                    [
                        probs.get("favor", 0.0),
                        probs.get("against", 0.0),
                        probs.get("neutral", 0.0),
                    ]
                )
            y_pred.append(pred)
            types.append(target_type)

    def _scope(name: str, yt: List[int], yp: List[int]) -> Dict[str, float]:
        if not yt:
            return {
                "%s_f1_macro" % name: 0.0,
                "%s_f1_favor" % name: 0.0,
                "%s_f1_against" % name: 0.0,
                "%s_f1_neutral" % name: 0.0,
                "%s_num" % name: 0,
            }
        per = _per_class_f1(yt, yp)
        present = sorted(set(yt) | set(yp))
        macro = sum(per[c] for c in present) / len(present) if present else 0.0
        return {
            "%s_f1_macro" % name: float(macro),
            "%s_f1_favor" % name: float(per[0]),
            "%s_f1_against" % name: float(per[1]),
            "%s_f1_neutral" % name: float(per[2]),
            "%s_num" % name: int(len(yt)),
        }

    metrics: Dict[str, Any] = {"teacher_files": len(files)}
    metrics.update(_scope("mixed", y_true, y_pred))
    for scope_name, keep in (("c", "c"), ("n", "n")):
        idx = [i for i, t in enumerate(types) if t == keep]
        metrics.update(_scope(scope_name, [y_true[i] for i in idx], [y_pred[i] for i in idx]))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline Qwen Instruct stance Teacher")
    p.add_argument("--data_path", required=True, help="CSD json with global_ids")
    p.add_argument("--output_dir", default="data/stance_teacher")
    p.add_argument("--provider", default="mock", choices=["mock", "heuristic", "local", "hf", "openai", "qwen", "vllm"])
    p.add_argument("--model", default="", help="Qwen3-8B local dir / HF id / served name")
    p.add_argument("--method", default="token", choices=["token", "json"], help="local: token logits; api: json")
    p.add_argument("--scope", default="last_sentence", choices=["last_sentence", "all"],
                   help="official CSD uses last_sentence")
    p.add_argument("--api_base", default="")
    p.add_argument("--api_key", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_conversations", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher = StanceTeacher(
        provider=args.provider,
        model=args.model,
        method=args.method,
        api_base=args.api_base,
        api_key=args.api_key,
        device=args.device,
        dtype=args.dtype,
        allow_download=args.allow_download,
        temperature=args.temperature,
        sleep_s=args.sleep,
    )
    records = _load_records(args.data_path)
    if args.max_conversations > 0:
        records = records[: args.max_conversations]
    n_ok = n_skip = 0
    for i, rec in enumerate(records, start=1):
        payload = generate_for_conversation(
            rec,
            teacher,
            skip_existing=not args.overwrite,
            output_dir=output_dir,
            scope=args.scope,
        )
        if payload is None:
            n_skip += 1
        else:
            n_ok += 1
        if i % 20 == 0 or i == len(records):
            logger.info(
                "progress %d/%d written=%d skipped=%d last_id=%s",
                i,
                len(records),
                n_ok,
                n_skip,
                rec.get("doc_id"),
            )
    quality = evaluate_teacher_dir(output_dir)
    (output_dir / "teacher_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Teacher last-sentence quality | Macro F1: %.2f | Favor: %.2f | Against: %.2f | Neutral: %.2f | n=%d",
        100.0 * quality.get("mixed_f1_macro", 0.0),
        100.0 * quality.get("mixed_f1_favor", 0.0),
        100.0 * quality.get("mixed_f1_against", 0.0),
        100.0 * quality.get("mixed_f1_neutral", 0.0),
        quality.get("mixed_num", 0),
    )
    logger.info("done written=%d skipped=%d quality=%s", n_ok, n_skip, output_dir / "teacher_quality.json")


if __name__ == "__main__":
    main()
