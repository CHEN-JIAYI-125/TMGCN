#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/9/18

"""Offline conversation-level embedding pipeline with Qwen3-Embedding-8B.

For every dialogue sample, build one stance-oriented prompt and encode it
into a single vector with a frozen Qwen3-Embedding model::

    Represent the stance expressed by the final utterance toward the target, considering the entire conversation context.

    Target:
    <target>

    Conversation:
    [1] <utt_1>
    [2] <utt_2>
    ...
    [N] <utt_N>

    Final utterance:
    <utt_N>

Row ``i`` of each output ``.npy`` corresponds to record ``i`` of the input
JSON file (original order preserved). Split files are encoded independently::

    <output_dir>/train.npy   [N_train, D] float32
    <output_dir>/dev.npy     [N_dev, D]   float32
    <output_dir>/test.npy    [N_test, D]  float32
    <output_dir>/<split>_ids.npy   record `id` field aligned with rows
    <output_dir>/summary.json      config + shapes + prompt template

Example (server)::

    python build_embedding_qwen.py \\
        --model_dir /xxxx/Qwen3-Embedding-8B \\
        --data_dir /path/to/json_dir \\
        --output_dir /path/to/qwen8b_conv_emb \\
        --batch_size 8 --max_length 2048 --dtype bf16

Notes on Qwen3-Embedding:
    * Decoder-only backbone -> use **last-token pooling** (not mean pooling).
    * Left padding is enforced so that the last position is always real text.
    * Embeddings are L2-normalized by default (standard for retrieval-style
      embedding models); disable with ``--no-normalize``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional on the server
    tqdm = None

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


PROMPT_HEADER = (
    "Represent the stance expressed by the final utterance toward the target, "
    "considering the entire conversation context."
)


def build_prompt(record: Dict[str, Any]) -> str:
    """Render one record into the stance prompt described in the docstring."""
    target = str(record.get("target", "") or "").strip()
    sentences: Sequence[str] = record.get("sentences", []) or []
    sentences = [str(s) for s in sentences]
    lines = [f"[{i + 1}] {s}" for i, s in enumerate(sentences)]
    final_utt = sentences[-1] if sentences else ""
    return (
        f"{PROMPT_HEADER}\n"
        f"\n"
        f"Target:\n"
        f"{target}\n"
        f"\n"
        f"Conversation:\n"
        f"{chr(10).join(lines)}\n"
        f"\n"
        f"Final utterance:\n"
        f"{final_utt}"
    )


# --------------------------------------------------------------------------- #
# Config / CLI
# --------------------------------------------------------------------------- #

@dataclass
class BuildConfig:
    model_dir: str = "/home/xiongjie/data/models/qwen3-embedding-8b"
    data_dir: str = "/home/xiongjie/data/csd/data/raw_with_all_labels_v3"
    output_dir: str = "/home/xiongjie/data/csd/data/raw_with_all_labels_v3/qwen3_embedding_8b"
    train_file: str = "/home/xiongjie/data/csd/data/raw_with_all_labels_v3/train_data.json"  # explicit path overrides auto-discovery under data_dir
    dev_file: str = "/home/xiongjie/data/csd/data/raw_with_all_labels_v3/dev_data.json"
    test_file: str = "/home/xiongjie/data/csd/data/raw_with_all_labels_v3/test_data.json"
    batch_size: int = 8
    max_length: int = 2048
    dtype: str = "bf16"  # auto / bf16 / fp16 / fp32
    device: str = "auto"  # auto / cpu / cuda / cuda:N
    device_map: str = "none"  # none / auto (multi-GPU sharding via accelerate)
    normalize: bool = True
    trust_remote_code: bool = True
    log_every: int = 20

    def to_json_string(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode one stance prompt per conversation with Qwen3-Embedding-8B."
    )
    d = BuildConfig()
    parser.add_argument("--model_dir", type=str, default=d.model_dir)
    parser.add_argument("--data_dir", type=str, default=d.data_dir)
    parser.add_argument("--output_dir", type=str, default=d.output_dir)
    parser.add_argument("--train_file", type=str, default=d.train_file,
                        help="explicit train JSON path; default: auto-discover '*train*.json' under --data_dir")
    parser.add_argument("--dev_file", type=str, default=d.dev_file,
                        help="explicit dev JSON path; default: auto-discover '*dev*.json' under --data_dir")
    parser.add_argument("--test_file", type=str, default=d.test_file,
                        help="explicit test JSON path; default: auto-discover '*test*.json' under --data_dir")
    parser.add_argument("--batch_size", type=int, default=d.batch_size)
    parser.add_argument("--max_length", type=int, default=d.max_length)
    parser.add_argument("--dtype", type=str, default=d.dtype,
                        choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=d.device, help="auto / cpu / cuda / cuda:N")
    parser.add_argument("--device_map", type=str, default=d.device_map,
                        help="none: single device; auto: accelerate sharding across GPUs")
    parser.add_argument("--normalize", dest="normalize", action="store_true",
                        help="L2-normalize embeddings (default: on)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="disable L2 normalization")
    parser.set_defaults(normalize=d.normalize)
    parser.add_argument("--trust_remote_code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=d.trust_remote_code)
    parser.add_argument("--log_every", type=int, default=d.log_every)
    return parser


def resolve_torch_dtype(name: str) -> Optional[torch.dtype]:
    mapping = {
        "auto": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unknown --dtype {name!r}; choices: {sorted(mapping)}")
    return mapping[name]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _autodiscover(data_dir: Path, keyword: str) -> Optional[Path]:
    candidates = sorted(data_dir.glob(f"*{keyword}*.json"))
    # Prefer canonical names first (train.json / train_data.json ...).
    for preferred in (f"{keyword}.json", f"{keyword}_data.json"):
        hit = data_dir / preferred
        if hit.exists():
            return hit
    return candidates[0] if candidates else None


def resolve_split_files(cfg: BuildConfig) -> Dict[str, Path]:
    explicit = {"train": cfg.train_file, "dev": cfg.dev_file, "test": cfg.test_file}
    data_dir = Path(cfg.data_dir)
    resolved: Dict[str, Path] = {}
    for split, given in explicit.items():
        if given:
            path = Path(given)
        else:
            if not data_dir.is_dir():
                raise FileNotFoundError(
                    f"--data_dir {str(data_dir)!r} does not exist; "
                    f"pass explicit --{split}_file instead."
                )
            found = _autodiscover(data_dir, split)
            if found is None:
                raise FileNotFoundError(
                    f"no *{split}*.json found under {data_dir}; "
                    f"pass explicit --{split}_file."
                )
            path = found
        if not path.exists():
            raise FileNotFoundError(f"{split} file not found: {path}")
        resolved[split] = path
    logger.info("split files: %s", {k: str(v) for k, v in resolved.items()})
    return resolved


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "samples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(f"{path} holds a JSON object without a record list; keys={list(data)}")
    if not isinstance(data, list):
        raise ValueError(f"{path} must hold a JSON list of records")
    logger.info("loaded %d record(s) from %s", len(data), path)
    return data


# --------------------------------------------------------------------------- #
# Encoding (last-token pooling for decoder-only Qwen3-Embedding)
# --------------------------------------------------------------------------- #

def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last *real* token per sequence; works for left or right padding."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden.shape[0]
    return last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]


@torch.no_grad()
def encode_prompts(
    prompts: List[str],
    tokenizer: Any,
    model: torch.nn.Module,
    encode_device: torch.device,
    batch_size: int,
    max_length: int,
    normalize: bool,
    log_every: int,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    iterator = range(0, len(prompts), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="encoding", unit="batch")
    for bi, start in enumerate(iterator):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(encode_device)
        attention_mask = encoded["attention_mask"].to(encode_device)
        hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emb = last_token_pool(hidden, attention_mask)
        if normalize:
            emb = F.normalize(emb.float(), p=2, dim=1)
        outputs.append(emb.float().cpu().numpy())
        if tqdm is None and (bi % max(1, log_every) == 0):
            logger.info("encoded %d/%d prompts", min(start + batch_size, len(prompts)), len(prompts))
    if not outputs:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def save_split(
    embeddings: np.ndarray,
    records: List[Dict[str, Any]],
    split: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    npy_path = output_dir / f"{split}.npy"
    with open(npy_path, "wb") as f:
        np.save(f, embeddings.astype(np.float32))
    ids = [rec.get("id", i) for i, rec in enumerate(records)]
    ids_path = output_dir / f"{split}_ids.npy"
    with open(ids_path, "wb") as f:
        np.save(f, np.asarray(ids))
    logger.info(
        "%s: wrote %s (%s) shape=%s | %s (%d ids)",
        split, npy_path, _human_size(npy_path.stat().st_size),
        tuple(embeddings.shape), ids_path, len(ids),
    )
    return npy_path, ids_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = build_parser().parse_args()
    cfg = BuildConfig(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        normalize=args.normalize,
        trust_remote_code=args.trust_remote_code,
        log_every=args.log_every,
    )
    logger.info("config:\n%s", cfg.to_json_string())
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = resolve_split_files(cfg)
    split_records = {split: load_records(path) for split, path in split_files.items()}
    split_prompts = {
        split: [build_prompt(rec) for rec in records]
        for split, records in split_records.items()
    }
    total = sum(len(v) for v in split_prompts.values())
    logger.info(
        "built %d prompt(s) (train=%d dev=%d test=%d); preview:\n%s",
        total,
        len(split_prompts.get("train", [])),
        len(split_prompts.get("dev", [])),
        len(split_prompts.get("test", [])),
        split_prompts[next(iter(split_prompts))][0] if total else "<empty>",
    )

    model_path = Path(cfg.model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"--model_dir {str(model_path)!r} does not exist on this machine. "
            "On the server pass the downloaded weights, e.g. /xxxx/Qwen3-Embedding-8B."
        )

    torch_dtype = resolve_torch_dtype(cfg.dtype)
    device = resolve_device(cfg.device)
    logger.info("using device=%s dtype=%s device_map=%s", device, cfg.dtype, cfg.device_map)

    logger.info("loading tokenizer/model from %s ...", model_path)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for last-token pooling

    load_kwargs: Dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if cfg.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(str(model_path), **load_kwargs)
    if cfg.device_map == "none":
        model.to(device)
        encode_device = device
    else:
        encode_device = getattr(model, "device", device)
    logger.info("model loaded in %.2fs (encode_device=%s)", time.time() - start, encode_device)

    shapes: Dict[str, List[int]] = {}
    counts: Dict[str, int] = {}
    for split in ("train", "dev", "test"):
        prompts = split_prompts.get(split, [])
        if not prompts:
            logger.warning("split %s is empty; skipping", split)
            continue
        logger.info("encoding split=%s (%d prompts) ...", split, len(prompts))
        t0 = time.time()
        embeddings = encode_prompts(
            prompts, tokenizer, model, encode_device,
            cfg.batch_size, cfg.max_length, cfg.normalize, cfg.log_every,
        )
        logger.info("split=%s encoded in %.2fs -> %s", split, time.time() - t0, tuple(embeddings.shape))
        save_split(embeddings, split_records[split], split, output_dir)
        shapes[split] = list(embeddings.shape)
        counts[split] = len(prompts)

    summary = {
        "config": asdict(cfg),
        "files": {k: str(v) for k, v in split_files.items()},
        "counts": counts,
        "shapes": shapes,
        "embedding_dim": int(next(iter(shapes.values()))[1]) if shapes else 0,
        "prompt_header": PROMPT_HEADER,
        "pooling": "last_token",
        "padding_side": "left",
        "row_alignment": "row i of <split>.npy == record i of the input JSON (order preserved); ids in <split>_ids.npy",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
    logger.info("summary written to %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
