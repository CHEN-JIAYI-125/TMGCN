#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线提取 Qwen3-Embedding 会话表示（不微调、不调付费 API）。

默认 unit=last_sentence：
  每段完整会话 → 1 个向量
  对应标签 = 该会话最后一句立场（官方 CSD 评价单元）
  不会把同一个向量复制给会话里的其它话语。

可选 unit=prefix：
  对每个 turn t 编码前缀 u_0..u_t，得到话语级可区分表示。

缓存目录（每个 split 一份）：
  embeddings.npy     [N, D] float32
  index.json         与行对齐的 id / 标签 / 文本元数据
  config.json        模型版本与提取配置
  length_stats.json  token 长度分布与溢出计数
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from conversation_text import iter_units

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("extract_conversation_embeddings")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "samples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError("%s is a JSON object without a record list; keys=%s" % (path, list(data)))
    if not isinstance(data, list):
        raise ValueError("%s must be a JSON list of CSD records" % path)
    return data


def expand_records(records: Sequence[Dict[str, Any]], unit: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for src_i, rec in enumerate(records):
        units = iter_units(rec, unit)
        for u in units:
            u["source_index"] = src_i
            rows.append(u)
    return rows


def dummy_encode(texts: Sequence[str], dim: int, seed: int) -> np.ndarray:
    """无模型时的确定性伪向量，只用于缓存/分类器通路测试。"""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        digest = hashlib.sha256(("%d||%s" % (seed, text)).encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "little") % (2**32)
        rng = np.random.RandomState(local_seed)
        vec = rng.normal(size=(dim,)).astype(np.float32)
        nrm = float(np.linalg.norm(vec))
        out[i] = vec / nrm if nrm > 0 else vec
    return out


def resolve_dtype(name: str):
    import torch

    mapping = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError("unknown --dtype %r" % name)
    return mapping[name]


def resolve_device(name: str):
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def count_tokens(tokenizer: Any, texts: Sequence[str], batch: int = 32) -> List[int]:
    lengths: List[int] = []
    for start in range(0, len(texts), batch):
        chunk = list(texts[start : start + batch])
        encoded = tokenizer(chunk, padding=False, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def length_report(lengths: Sequence[int], max_length: int) -> Dict[str, Any]:
    arr = np.asarray(lengths, dtype=np.int64)
    if arr.size == 0:
        return {"n": 0, "max_length": max_length}
    overflow_idx = np.where(arr > max_length)[0].tolist()
    return {
        "n": int(arr.size),
        "max_length": int(max_length),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "n_overflow": int(len(overflow_idx)),
        "overflow_ratio": float(len(overflow_idx) / arr.size),
        "overflow_row_indices": overflow_idx[:50],
    }


def chunk_encode_one(text: str, encode_fn, max_length: int, tokenizer: Any) -> np.ndarray:
    """超长会话分块编码后均值池化。不丢弃数据。"""
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if not ids:
        raise ValueError("empty tokenization")
    pieces = []
    for start in range(0, len(ids), max_length):
        sub_ids = ids[start : start + max_length]
        sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True)
        pieces.append(encode_fn([sub_text])[0])
    stacked = np.stack(pieces, axis=0)
    vec = stacked.mean(axis=0)
    nrm = float(np.linalg.norm(vec))
    return vec / nrm if nrm > 0 else vec


def load_st_model(model_name: str, device: str, torch_dtype, trust_remote_code: bool):
    from sentence_transformers import SentenceTransformer

    kwargs: Dict[str, Any] = {
        "device": device,
        "trust_remote_code": trust_remote_code,
        "tokenizer_kwargs": {"padding_side": "left"},
    }
    model_kwargs: Dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    logger.info("loading SentenceTransformer from %s on %s ...", model_name, device)
    t0 = time.time()
    model = SentenceTransformer(model_name, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("model loaded in %.1fs", time.time() - t0)
    return model


def st_encode_fn(model, normalize: bool, batch_size: int):
    def _encode(texts: Sequence[str]) -> np.ndarray:
        vec = model.encode(
            list(texts),
            batch_size=max(1, batch_size),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype=np.float32)

    return _encode


def load_hf_model(model_name: str, device: str, torch_dtype, trust_remote_code: bool):
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("sentence-transformers 不可用，回退 transformers AutoModel: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def hf_encode_fn(tokenizer, model, device: str, max_length: int, normalize: bool):
    import torch
    import torch.nn.functional as F

    def last_token_pool(last_hidden, attention_mask):
        left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden[:, -1]
        seq_len = attention_mask.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_len]

    def _encode(texts: Sequence[str]) -> np.ndarray:
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
            emb = last_token_pool(hidden, encoded["attention_mask"]).float()
            if normalize:
                emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    return _encode


def encode_rows(
    texts: List[str],
    encode_fn,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    overflow: str,
    lengths: Optional[List[int]],
) -> Tuple[np.ndarray, int]:
    n_chunked = 0
    vectors: List[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="encoding", unit="batch")
    for start in iterator:
        batch_texts = texts[start : start + batch_size]
        try:
            if overflow == "chunk" and lengths is not None:
                piece = []
                for offset, text in enumerate(batch_texts):
                    row = start + offset
                    if lengths[row] > max_length:
                        piece.append(chunk_encode_one(text, encode_fn, max_length, tokenizer))
                        n_chunked += 1
                    else:
                        piece.append(encode_fn([text])[0])
                vec = np.stack(piece, axis=0)
            else:
                vec = encode_fn(batch_texts)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                logger.error(
                    "CUDA 显存不足（已处理到第 %d/%d 条）。数据没有被静默截断。建议："
                    " 1) --batch_size 1  2) --max_length 1024  3) --dtype fp16 "
                    " 4) 先用 Qwen/Qwen3-Embedding-0.6B  5) 换更大显存卡。",
                    start,
                    len(texts),
                )
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                raise
            raise
        if vec.ndim != 2 or vec.shape[0] != len(batch_texts):
            raise RuntimeError(
                "encoder 输出形状异常: got %s, expected [%d, D]" % (tuple(vec.shape), len(batch_texts))
            )
        vectors.append(np.asarray(vec, dtype=np.float32))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), n_chunked
    return np.concatenate(vectors, axis=0), n_chunked


def save_split_cache(
    output_dir: Path,
    split: str,
    embeddings: np.ndarray,
    index_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    stats: Dict[str, Any],
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    if embeddings.shape[0] != len(index_rows):
        raise RuntimeError(
            "行数不一致: embeddings %d vs index %d" % (embeddings.shape[0], len(index_rows))
        )
    npy_path = split_dir / "embeddings.npy"
    np.save(npy_path, embeddings.astype(np.float32))
    slim_index = []
    for row in index_rows:
        slim_index.append(
            {
                "conversation_id": row["conversation_id"],
                "source_index": row["source_index"],
                "turn_index": row["turn_index"],
                "label": row["label"],
                "target": row["target"],
                "target_type": row["target_type"],
                "n_turns": row["n_turns"],
                "n_tokens": row.get("n_tokens"),
            }
        )
    with open(split_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(slim_index, f, ensure_ascii=False, indent=2)
    with open(split_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(split_dir / "length_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(
        "saved %s: %s shape=%s dim=%s n=%d",
        split,
        npy_path,
        tuple(embeddings.shape),
        embeddings.shape[1] if embeddings.ndim == 2 and embeddings.size else 0,
        len(slim_index),
    )


def load_split_cache(split_dir: Path) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    npy_path = split_dir / "embeddings.npy"
    idx_path = split_dir / "index.json"
    cfg_path = split_dir / "config.json"
    if not npy_path.is_file() or not idx_path.is_file():
        raise FileNotFoundError("缺少缓存文件: %s 或 %s" % (npy_path, idx_path))
    embeddings = np.load(npy_path)
    with open(idx_path, "r", encoding="utf-8") as f:
        index_rows = json.load(f)
    config = {}
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    if embeddings.ndim != 2:
        raise ValueError("embeddings 必须是 [N, D]，得到 %s" % (embeddings.shape,))
    if embeddings.shape[0] != len(index_rows):
        raise ValueError(
            "数量不一致: embeddings 行=%d index=%d dir=%s"
            % (embeddings.shape[0], len(index_rows), split_dir)
        )
    expected_dim = config.get("embedding_dim")
    if expected_dim is not None and int(expected_dim) != int(embeddings.shape[1]):
        raise ValueError(
            "维度不一致: config.embedding_dim=%s 实际=%s dir=%s"
            % (expected_dim, embeddings.shape[1], split_dir)
        )
    return embeddings.astype(np.float32), index_rows, config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Qwen3 conversation embeddings for CSD baseline")
    p.add_argument("--train_file", type=str, default="")
    p.add_argument("--dev_file", type=str, default="")
    p.add_argument("--test_file", type=str, default="")
    p.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="若未给分文件，则在此目录找 train_data.json / dev_data.json / test_data.json",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--unit", type=str, default="last_sentence", choices=["last_sentence", "prefix"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:N")
    p.add_argument("--overflow", type=str, default="truncate", choices=["error", "truncate", "chunk"])
    p.add_argument("--max_records", type=int, default=0, help=">0 时每个 split 只取前 K 条，冒烟测试用")
    p.add_argument("--dummy", action="store_true", help="不加载大模型，写确定性伪向量（只测通路）")
    p.add_argument("--dummy_dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--verify_only", action="store_true", help="只校验已有缓存，不重新提取")
    return p.parse_args()


def resolve_split_files(args: argparse.Namespace) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    explicit = {"train": args.train_file, "dev": args.dev_file, "test": args.test_file}
    data_dir = Path(args.data_dir) if args.data_dir else None
    for split, given in explicit.items():
        if given:
            path = Path(given)
        elif data_dir is not None:
            path = data_dir / ("%s_data.json" % split)
        else:
            raise ValueError("请提供 --%s_file 或 --data_dir" % split)
        if not path.is_file():
            raise FileNotFoundError("%s split 不存在: %s" % (split, path))
        out[split] = path
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        for split in ("train", "dev", "test"):
            split_dir = output_dir / split
            emb, index_rows, config = load_split_cache(split_dir)
            logger.info(
                "OK %s n=%d dim=%d unit=%s model=%s",
                split,
                emb.shape[0],
                emb.shape[1],
                config.get("unit"),
                config.get("model"),
            )
        return

    split_files = resolve_split_files(args)
    device = "cpu" if args.dummy else resolve_device(args.device)
    torch_dtype = None if args.dummy else resolve_dtype(args.dtype)
    normalize = not args.no_normalize

    tokenizer = None
    encode_fn = None
    backend = "dummy"
    embedding_dim = int(args.dummy_dim)
    if args.dummy:
        encode_fn = lambda texts: dummy_encode(texts, args.dummy_dim, args.seed)
        logger.warning("DUMMY encoder 开启：结果不能当真实实验")
    else:
        try:
            model = load_st_model(args.model, device, torch_dtype, args.trust_remote_code)
            try:
                model.max_seq_length = int(args.max_length)
            except Exception:
                pass
            encode_fn = st_encode_fn(model, normalize, args.batch_size)
            tokenizer = model.tokenizer
            backend = "sentence_transformers"
            try:
                embedding_dim = int(model.get_sentence_embedding_dimension())
            except Exception:
                embedding_dim = -1
        except ImportError:
            tokenizer, hf_model = load_hf_model(args.model, device, torch_dtype, args.trust_remote_code)
            encode_fn = hf_encode_fn(tokenizer, hf_model, device, args.max_length, normalize)
            backend = "transformers"
            embedding_dim = int(getattr(hf_model.config, "hidden_size", -1))

    global_config = {
        "model": "dummy" if args.dummy else args.model,
        "backend": backend,
        "unit": args.unit,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "device": device,
        "overflow": args.overflow,
        "normalize": normalize,
        "seed": args.seed,
        "dummy": bool(args.dummy),
        "embedding_dim": embedding_dim,
        "label_granularity": (
            "last_sentence_of_conversation"
            if args.unit == "last_sentence"
            else "per_utterance_via_causal_prefix"
        ),
        "embedding_granularity": (
            "one_vector_per_full_conversation"
            if args.unit == "last_sentence"
            else "one_vector_per_utterance_prefix"
        ),
    }

    for split, path in split_files.items():
        records = load_json_records(path)
        if args.max_records > 0:
            records = records[: args.max_records]
            logger.info("%s: truncated to first %d records for smoke test", split, len(records))
        rows = expand_records(records, args.unit)
        texts = [r["text"] for r in rows]
        logger.info("%s: %d json records -> %d embedding rows (unit=%s)", split, len(records), len(rows), args.unit)

        lengths: Optional[List[int]] = None
        stats: Dict[str, Any] = {"n": len(texts), "max_length": args.max_length}
        if tokenizer is not None:
            lengths = count_tokens(tokenizer, texts)
            for row, ntok in zip(rows, lengths):
                row["n_tokens"] = int(ntok)
            stats = length_report(lengths, args.max_length)
            logger.info("%s length stats: %s", split, json.dumps({k: stats[k] for k in stats if k != "overflow_row_indices"}))
            if stats["n_overflow"] > 0 and args.overflow == "error":
                logger.error(
                    "%s 有 %d 条超过 max_length=%d（max=%d）。"
                    "不会静默截断。改用 --overflow truncate（保留尾部）或 chunk，或增大 --max_length。",
                    split,
                    stats["n_overflow"],
                    args.max_length,
                    stats["max"],
                )
                raise SystemExit(2)
            if stats["n_overflow"] > 0 and args.overflow == "truncate":
                logger.warning(
                    "%s: %d 条将按 max_length=%d 从左侧截断（保留会话尾部/最后一句）",
                    split,
                    stats["n_overflow"],
                    args.max_length,
                )

        embeddings, n_chunked = encode_rows(
            texts,
            encode_fn,
            tokenizer,
            batch_size=max(1, args.batch_size),
            max_length=args.max_length,
            overflow=args.overflow,
            lengths=lengths,
        )
        stats["n_chunked"] = n_chunked
        if embeddings.shape[0] != len(rows):
            raise RuntimeError("编码行数与 index 不一致")
        if embedding_dim > 0 and embeddings.shape[1] != embedding_dim:
            logger.warning("config dim=%s 实际 dim=%s，以实际为准", embedding_dim, embeddings.shape[1])
            global_config["embedding_dim"] = int(embeddings.shape[1])
        elif embedding_dim <= 0:
            global_config["embedding_dim"] = int(embeddings.shape[1])

        cfg_this = dict(global_config)
        cfg_this["split"] = split
        cfg_this["source_file"] = str(path.resolve())
        cfg_this["n_json_records"] = len(records)
        cfg_this["n_rows"] = len(rows)
        save_split_cache(output_dir, split, embeddings, rows, cfg_this, stats)

    logger.info("done. output_dir=%s", output_dir.resolve())


if __name__ == "__main__":
    main()
