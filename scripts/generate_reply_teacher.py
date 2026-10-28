#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线生成 LLM Reply Teacher 的 soft 回复分布。

读取 ZS-CSD 对话数据（含 sentences / speakers / target / id），对每个
conversation 中的每个当前话语 u_t，只把「target + u_1..u_t」交给 LLM，
禁止看见未来话语。LLM 输出 u_t 对 ROOT 以及所有历史话语的 soft 回复概率。

每个 conversation 保存一个 JSON：

    data/reply_teacher/conversation_<id>.json

文件内容是该对话所有 turn 的列表，单条格式：

    {
      "conversation_id": 123,
      "utterance_id": 2,
      "reply_distribution": {"ROOT": 0.05, "0": 0.10, "1": 0.85}
    }

约束：
1. 必须包含 ROOT（不回复任何历史、开启新线程 / 面向话题）。
2. 只能选择当前话语之前的 utterances（键为 "0".."t-1"）。
3. 不允许看到未来 utterance。
4. 输出是 soft score，不是 hard label；读取后会做非负归一化。

用法示例
--------
# 1) 不调用 LLM，用近因启发式生成 soft 分布（打通数据管线 / debug）
python scripts/generate_reply_teacher.py \\
    --data_path /path/to/train_data.json \\
    --output_dir data/reply_teacher \\
    --provider mock

# 2) 本地 Qwen3-8B-Instruct（生成回复结构，不预测立场）
python scripts/generate_reply_teacher.py \\
    --data_path /path/to/train_data.json \\
    --output_dir data/reply_teacher \\
    --provider local \\
    --model /path/to/Qwen3-8B \\
    --device cuda:0

# 3) OpenAI 兼容接口（vLLM / DashScope）
python scripts/generate_reply_teacher.py \\
    --data_path /path/to/train_data.json \\
    --output_dir data/reply_teacher \\
    --provider openai \\
    --model Qwen3-8B \\
    --api_base http://127.0.0.1:8000/v1

对 train / dev / test 各跑一次，写到同一个 output_dir 即可
（conversation_id 不跨 split 冲突时文件名即 id）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("generate_reply_teacher")


SYSTEM_PROMPT = (
    "You are an expert conversational discourse analyst. "
    "Your only job is to estimate SOFT reply-structure probabilities: "
    "which previous utterance the CURRENT utterance is most likely responding to. "
    "You never predict stance labels."
)


def build_user_prompt(target: str, utterances: Sequence[Dict[str, Any]], t: int) -> str:
    """构造只含 u_0..u_t 的因果 prompt（不含未来）。"""
    lines = [
        "Estimate the reply-structure distribution of the CURRENT utterance.",
        "",
        "Target (conversation topic):",
        str(target).strip() or "(none)",
        "",
        "Conversation history visible to you (DO NOT use any later utterance):",
        "",
    ]
    for i in range(t + 1):
        u = utterances[i]
        marker = "  [CURRENT]" if i == t else ""
        lines.append(f"u{i} {u['speaker']}:{marker}")
        lines.append(str(u["text"]).strip())
        lines.append("")
    hist_keys = ", ".join(["ROOT"] + [f'"{j}"' for j in range(t)])
    lines.extend(
        [
            f"CURRENT utterance is u{t}.",
            "",
            "Options:",
            "- ROOT: u_t does not reply to any previous utterance "
            "(new thread, addresses the audience/topic, or opens the dialogue).",
            f"- Integer keys 0..{t - 1}: u_t is a reply to that previous utterance."
            if t > 0
            else "- There is no previous utterance; ROOT must get probability 1.0.",
            "",
            "Hard constraints:",
            "1. You MUST include ROOT.",
            "2. You may ONLY choose ROOT"
            + (f" and utterances u0..u{t - 1}." if t > 0 else "."),
            "3. You MUST NOT use future utterances (they are not shown).",
            "4. Output SOFT scores (all values >= 0, will be normalized to sum to 1). "
            "Do not output a single hard label.",
            "5. Typical conversations concentrate most mass on one or two antecedents, "
            "with a small residual on ROOT and other history.",
            "",
            "Return JSON only, no markdown, no commentary. Schema:",
            f'{{"reply_distribution": {{ {hist_keys} }}}}',
        ]
    )
    return "\n".join(lines)


def _load_json_records(raw_path: str) -> List[Dict[str, Any]]:
    """兼容单文件 JSON 数组，或目录下全部 *.json。字段与 TMGCN raw 一致。"""
    path = Path(raw_path)
    if path.is_dir():
        json_paths = sorted(path.glob("*.json"))
    elif path.is_file():
        json_paths = [path]
    else:
        raise FileNotFoundError(f"数据路径不存在: {raw_path}")
    if not json_paths:
        raise FileNotFoundError(f"未找到 JSON: {raw_path}")

    records: List[Dict[str, Any]] = []
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = items.get("data", items.get("records", [items]))
        for item in items:
            sentences = list(item.get("sentences", []) or [])
            if not sentences:
                continue
            speakers = list(item.get("speakers", []) or [])
            utts = []
            for i, text in enumerate(sentences):
                spk = speakers[i] if i < len(speakers) else "UNK"
                if isinstance(spk, int) or (isinstance(spk, str) and str(spk).isdigit()):
                    spk_name = f"speaker_{spk}"
                else:
                    spk_name = str(spk)
                utts.append({"speaker": spk_name, "text": text})
            records.append(
                {
                    "conversation_id": int(item.get("id", item.get("doc_id", item.get("conversation_id", -1)))),
                    "target": item.get("target", ""),
                    "utterances": utts,
                    "source": jp.name,
                }
            )
    logger.info("从 %s 读入 %d 段对话", raw_path, len(records))
    return records


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """从模型输出中抠出 JSON 对象（容忍 markdown 代码块）。"""
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


def normalize_reply_distribution(raw: Dict[str, Any], t: int) -> Dict[str, float]:
    """强制合法 support：ROOT + 0..t-1，非负、求和为 1。未来键全部丢弃。"""
    src = raw.get("reply_distribution", raw) if isinstance(raw, dict) else {}
    if not isinstance(src, dict):
        src = {}
    keys = ["ROOT"] + [str(j) for j in range(t)]
    scores: Dict[str, float] = {}
    for k in keys:
        val = src.get(k, src.get(k.lower()) if k == "ROOT" else None)
        try:
            scores[k] = max(0.0, float(val))
        except (TypeError, ValueError):
            scores[k] = 0.0
    total = sum(scores.values())
    if total <= 0.0:
        # 模型失败时的弱先验：近因 + 小 ROOT，而不是 one-hot
        scores = _recency_prior(t)
        total = sum(scores.values())
    return {k: scores[k] / total for k in keys}


def _recency_prior(t: int, root_mass: float = 0.08, recency_temp: float = 0.7) -> Dict[str, float]:
    """近因启发式 soft 分布：越靠近当前话语的历史权重越高。

    用于 mock provider，以及 LLM 解析失败时的回退，保证永远是 soft score。
    """
    if t <= 0:
        return {"ROOT": 1.0}
    weights = []
    for j in range(t):
        # j 越接近 t-1，分数越大
        gap = float((t - 1) - j)
        weights.append(pow(2.718281828, -gap / max(1e-6, recency_temp)))
    s = sum(weights)
    dist = {"ROOT": float(root_mass)}
    remain = 1.0 - float(root_mass)
    for j, w in enumerate(weights):
        dist[str(j)] = remain * (w / s)
    return dist


def _openai_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 120,
    temperature: float = 0.4,
) -> str:
    """最小依赖的 OpenAI-compatible Chat Completions 调用。"""
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
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    try:
        return body["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"unexpected LLM response: {body}") from exc


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _patch_torch_distributed_tensor() -> None:
    """Qwen3 的 transformers 会 import torch.distributed.tensor（PyTorch>=2.5）。

    2.4 里同名 API 在 torch.distributed._tensor，先做别名；再旧就只能升级 torch
    或改用 Qwen2.5-7B-Instruct。
    """
    import sys
    import types

    try:
        import torch.distributed.tensor  # noqa: F401
        return
    except Exception:
        pass
    try:
        import torch.distributed._tensor as _dt  # type: ignore

        sys.modules["torch.distributed.tensor"] = _dt
        return
    except Exception:
        pass
    dummy = types.ModuleType("torch.distributed.tensor")
    dummy.DTensor = type("DTensor", (), {})
    dummy.Shard = type("Shard", (), {})
    dummy.Replicate = type("Replicate", (), {})
    dummy.Partial = type("Partial", (), {})
    dummy.Placement = type("Placement", (), {})
    sys.modules.setdefault("torch.distributed.tensor", dummy)


class LocalQwenReplyTeacher:
    """本地 Qwen3-Instruct：只生成回复结构 JSON，不预测立场。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bf16",
        allow_download: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.4,
    ):
        import torch

        _patch_torch_distributed_tensor()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "当前环境加载 Qwen3 失败（需要较新的 PyTorch，一般 >= 2.5）。\n"
                "先查版本: python3.9 -c \"import torch,transformers; print(torch.__version__, transformers.__version__)\"\n"
                "可选: 1) 升级 torch  2) 改用 --model Qwen/Qwen2.5-7B-Instruct --allow-download\n"
                "原始错误: %s" % exc
            ) from exc

        self.torch = torch
        local_only = not allow_download
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_only, trust_remote_code=True
        )
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "auto" else None,
                local_files_only=local_only,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "加载 %s 失败。若仍是 torch.distributed.tensor，请升级 PyTorch>=2.5，"
                "或改用 Qwen/Qwen2.5-7B-Instruct。原始错误: %s" % (model_path, exc)
            ) from exc
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages: List[Dict[str, str]]) -> str:
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
        return self.tokenizer.decode(out[0][inputs["input_ids"].size(1) :], skip_special_tokens=True)


class LLMReplyTeacher:
    """离线 Teacher：对单个 (conversation, t) 查询回复分布。"""

    def __init__(
        self,
        provider: str,
        model: str = "",
        api_base: str = "",
        api_key: str = "",
        temperature: float = 0.4,
        max_retries: int = 3,
        sleep_s: float = 0.5,
        device: str = "cuda:0",
        dtype: str = "bf16",
        allow_download: bool = False,
    ):
        self.provider = str(provider).lower()
        self.model = model
        self.api_base = api_base or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.sleep_s = float(sleep_s)
        self.local: Optional[LocalQwenReplyTeacher] = None
        if self.provider in ("local", "hf", "huggingface"):
            if not self.model:
                raise ValueError("provider=local 时必须指定 --model（Qwen3-8B 本地目录或 HF id）")
            self.local = LocalQwenReplyTeacher(
                model_path=self.model,
                device=device,
                dtype=dtype,
                allow_download=allow_download,
                temperature=self.temperature,
            )
        elif self.provider in ("openai", "qwen", "vllm") and not self.model:
            raise ValueError("provider=%s 时必须指定 --model" % self.provider)

    def infer(self, target: str, utterances: Sequence[Dict[str, Any]], t: int) -> Dict[str, float]:
        if t <= 0:
            return {"ROOT": 1.0}
        if self.provider in ("mock", "heuristic", "none"):
            return _recency_prior(t)

        prompt = build_user_prompt(target, utterances, t)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.local is not None:
                    text = self.local.generate(messages)
                else:
                    text = _openai_chat(
                        self.api_base,
                        self.api_key,
                        self.model,
                        messages,
                        temperature=self.temperature,
                    )
                obj = _extract_json_obj(text)
                return normalize_reply_distribution(obj, t)
            except Exception as exc:
                last_err = exc
                logger.warning("LLM 调用失败 attempt=%d/%d: %s", attempt, self.max_retries, exc)
                time.sleep(self.sleep_s * attempt)
        logger.error("LLM 全部重试失败，回退近因先验: %s", last_err)
        return _recency_prior(t)


def conversation_out_path(output_dir: Path, conversation_id: Any) -> Path:
    return output_dir / f"conversation_{conversation_id}.json"


def generate_for_conversation(
    rec: Dict[str, Any],
    teacher: LLMReplyTeacher,
    skip_existing: bool,
    output_dir: Path,
) -> Optional[List[Dict[str, Any]]]:
    cid = rec["conversation_id"]
    out_path = conversation_out_path(output_dir, cid)
    if skip_existing and out_path.is_file():
        return None
    utts = rec["utterances"]
    rows: List[Dict[str, Any]] = []
    for t in range(len(utts)):
        dist = teacher.infer(rec["target"], utts, t)
        dist = normalize_reply_distribution({"reply_distribution": dist}, t)
        rows.append(
            {
                "conversation_id": cid,
                "utterance_id": t,
                "target": rec["target"],
                "reply_distribution": dist,
            }
        )
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线生成 LLM Reply Teacher 分布")
    parser.add_argument("--data_path", required=True, help="ZS-CSD json 文件或目录")
    parser.add_argument("--output_dir", default="data/reply_teacher", help="Teacher JSON 输出目录")
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "heuristic", "local", "hf", "openai", "qwen", "vllm"],
        help="mock=近因先验；local=本地 Qwen Instruct；其余走 OpenAI 兼容 Chat API",
    )
    parser.add_argument("--model", default="", help="Qwen3-8B 本地目录 / HF id / 服务名")
    parser.add_argument("--api_base", default="", help="默认 OPENAI_BASE_URL 或 https://api.openai.com/v1")
    parser.add_argument("--api_key", default="", help="默认 OPENAI_API_KEY / DASHSCOPE_API_KEY")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.4, help="LLM 采样温度（>0 以得到 soft 分数）")
    parser.add_argument("--max_conversations", type=int, default=0, help=">0 时只处理前 K 段，调试用")
    parser.add_argument("--sleep", type=float, default=0.2, help="两次 LLM 调用之间的间隔秒")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 conversation_*.json")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = LLMReplyTeacher(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        sleep_s=args.sleep,
        device=args.device,
        dtype=args.dtype,
        allow_download=args.allow_download,
    )
    records = _load_json_records(args.data_path)
    if args.max_conversations > 0:
        records = records[: args.max_conversations]

    n_ok = n_skip = 0
    for i, rec in enumerate(records, start=1):
        rows = generate_for_conversation(
            rec,
            teacher,
            skip_existing=not args.overwrite,
            output_dir=output_dir,
        )
        if rows is None:
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
                rec["conversation_id"],
            )
        if args.provider not in ("mock", "heuristic") and rows is not None:
            time.sleep(args.sleep)

    logger.info(
        "完成 Teacher 生成: written=%d skipped=%d output_dir=%s",
        n_ok,
        n_skip,
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
