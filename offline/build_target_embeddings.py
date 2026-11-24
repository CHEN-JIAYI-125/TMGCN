#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/6/8

"""Offline utterance embedding pipeline for the graph-based CSD model.

Three stages, runnable together (`--stage all`) or separately:

Stage `dedup`
-------------
1. Iterate over every JSON file in `data_dir` in sorted order.
2. Walk every sample's `sentences` field; for each sentence assign a
   global_id. Identical sentences are globally deduplicated, so the same
   sentence text always maps to the same global_id (first occurrence wins).
3. Write a copy of each input JSON with an extra `global_ids` field aligned
   with `sentences` (same length), so downstream code can look up each
   utterance's embedding by row index.
4. Write `sentence.csv` with two columns (`global_id, sentence`), one row per
   unique sentence, ordered by global_id.

Stage `embed`
-------------
1. Read `sentence.csv`.
2. Encode each unique sentence with a frozen PLM (mean pooling of the last
   hidden state).
3. Save `embeddings.npy` of shape [N, D]; row index == global_id.

Stage `targets`
---------------
1. Read the target knowledge template JSON (one entry per target with a
   stance-neutral `description`, a `favor_reason` and an `against_reason`).
2. Encode, with the SAME frozen PLM and pooling as `embed`, four texts per
   target: the raw target text itself plus the three stance views.
3. Save `target_knowledge.npy` of shape [T, 4, D] (row order:
   [raw target text; background; favor; against]) and
   `target_knowledge_index.json` mapping target text -> row. These feed the
   target-side stance semantic grounding module of the v5 tracer.

Output layout under `output_dir`:
    raw_with_global_ids/*.json   original records + `global_ids`
    sentence.csv                 global_id, sentence
    embeddings.npy               [N, D] float32
    target_knowledge.npy         [T, 4, D] float32
    target_knowledge_index.json  {target_text: row}
    summary.json                 config + stats
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    data_dir: str = r"/home/xiongjie/data/csd/data/raw_with_all_labels_v2"
    bert_dir: str = "/home/xiongjie/data/models/chinese-roberta-wwm-ext/"  # path to chinese-roberta-wwm-ext or any AutoModel checkpoint
    output_dir: str = "/home/xiongjie/data/csd/data/raw_with_all_labels_v3"
    stage: str = "all"  # dedup / embed / targets / all
    batch_size: int = 64
    max_length: int = 128
    device: str = "auto"
    normalize: bool = False
    fp16: bool = False
    target_template: str = str(
        Path(__file__).resolve().parent.parent / "datasets" / "target_knowledge_template.json"
    )

    def to_json_string(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline utterance embeddings with a frozen PLM.")
    parser.add_argument("--data_dir", type=str, default=BuildConfig.data_dir)
    parser.add_argument("--bert_dir", type=str, default=BuildConfig.bert_dir)
    parser.add_argument("--output_dir", type=str, default=BuildConfig.output_dir)
    parser.add_argument("--stage", type=str, default=BuildConfig.stage, choices=["dedup", "embed", "targets", "all"])
    parser.add_argument("--batch_size", type=int, default=BuildConfig.batch_size)
    parser.add_argument("--max_length", type=int, default=BuildConfig.max_length)
    parser.add_argument("--device", type=str, default=BuildConfig.device, help="cpu / cuda:N / auto")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--target_template", type=str, default=BuildConfig.target_template,
        help="target knowledge template JSON (description / favor_reason / against_reason per target)",
    )
    return parser


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Stage: dedup
# --------------------------------------------------------------------------- #


def load_raw_records(data_dir: str) -> List[Tuple[Path, List[Dict[str, Any]]]]:
    """Load every JSON file in `data_dir`; returns (path, records) per file."""
    data_dir = Path(data_dir)
    json_paths = sorted(data_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found under {data_dir}")

    loaded: List[Tuple[Path, List[Dict[str, Any]]]] = []
    for path in json_paths:
        if path.name == "label.json":
            continue
        with open(path, "r", encoding="utf-8") as f:
            loaded.append((path, json.load(f)))
    logger.info("loaded %d JSON file(s) from %s", len(loaded), data_dir)
    return loaded


def assign_global_ids(
    files: List[Tuple[Path, List[Dict[str, Any]]]],
) -> Tuple[List[Tuple[Path, List[Dict[str, Any]]]], List[Tuple[int, str]]]:
    """Assign global_ids to every sentence; dedup identical sentence texts.

    Returns:
        files: each record extended with a `global_ids` field aligned with `sentences`.
        sentence_rows: list of (global_id, sentence), one per unique sentence.
    """
    text_to_global_id: Dict[str, int] = {}
    sentence_rows: List[Tuple[int, str]] = []
    next_global_id = 0

    for path, records in files:
        for record in records:
            sentences = record.get("sentences", []) or []
            global_ids = []
            for sentence in sentences:
                if sentence not in text_to_global_id:
                    text_to_global_id[sentence] = next_global_id
                    sentence_rows.append((next_global_id, sentence))
                    next_global_id += 1
                global_ids.append(text_to_global_id[sentence])
            record["global_ids"] = global_ids

    logger.info(
        "assigned %d global_ids over %d unique sentences (deduped %d duplicates)",
        next_global_id,
        len(sentence_rows),
        sum(len(rec.get("sentences", [])) for _, recs in files for rec in recs) - len(sentence_rows),
    )
    return files, sentence_rows


def write_dedup_outputs(
    files: List[Tuple[Path, List[Dict[str, Any]]]],
    sentence_rows: List[Tuple[int, str]],
    output_dir: Path,
) -> None:
    raw_dir = output_dir / "raw_with_global_ids"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for src_path, records in files:
        dest_path = raw_dir / src_path.name
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("wrote %s (%d records)", dest_path, len(records))

    sentence_rows = sorted(sentence_rows, key=lambda x: x[0])
    csv_path = output_dir / "sentence.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_id", "sentence"])
        writer.writerows(sentence_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(sentence_rows))


# --------------------------------------------------------------------------- #
# Stage: embed
# --------------------------------------------------------------------------- #


def read_sentence_csv(csv_path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[:2] != ["global_id", "sentence"]:
            raise ValueError(f"unexpected header in {csv_path}: {header}")
        for line in reader:
            if len(line) < 2:
                continue
            rows.append((int(line[0]), line[1]))
    return rows


@torch.no_grad()
def encode_unique_sentences(
    sentences: List[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
    fp16: bool,
) -> np.ndarray:
    """Encode a list of unique sentences; mean pooling of the last hidden state."""
    embeddings: List[np.ndarray] = []
    model.eval()
    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        encoded = tokenizer(
            sentences[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if fp16 and device.type == "cuda":
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden = outputs.last_hidden_state  # [B, L, D]
        mask = attention_mask.unsqueeze(-1).float()
        emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embeddings.append(emb.float().cpu().numpy())

    return (
        np.concatenate(embeddings, axis=0)
        if embeddings
        else np.zeros((0,), dtype=np.float32)
    )


def write_embed_outputs(
    embeddings: np.ndarray,
    cfg: BuildConfig,
    num_unique: int,
    elapsed: float,
) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_path = output_dir / "embeddings.npy"
    with open(embed_path, "wb") as f:
        np.save(f, embeddings.astype(np.float32))

    summary = {
        "config": asdict(cfg),
        "num_unique_sentences": num_unique,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "elapsed_seconds": round(elapsed, 3),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)

    logger.info(
        "embeddings written to %s (%s) | shape=%s | dim=%d",
        embed_path,
        _human_size(embed_path.stat().st_size),
        embeddings.shape,
        summary["embedding_dim"],
    )
    logger.info("summary written to %s", output_dir / "summary.json")


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


# --------------------------------------------------------------------------- #
# Stage: targets (target-side stance semantic grounding prior)
# --------------------------------------------------------------------------- #


def load_target_template(template_path: str) -> List[Tuple[str, List[str]]]:
    """Read the target knowledge template JSON.

    Each entry holds a stance-neutral `description`, a `favor_reason` and an
    `against_reason` for one target. Returns (target_text, views) with views
    ordered [raw target text; background; favor; against] -- row 0 is the
    un-grounded target representation, rows 1..3 form K = [k^B; k^F; k^A].
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)
    if not isinstance(template, dict) or not template:
        raise ValueError(f"{template_path} must be a non-empty JSON object keyed by target text")

    rows: List[Tuple[str, List[str]]] = []
    for target, info in template.items():
        if not isinstance(info, dict):
            raise ValueError(f"target {target!r} in {template_path} must map to an object")
        background = str(info.get("description", "")).strip()
        # background_enriched = f"关于{target}，中立方通常认为：{background}"
        favor = str(info.get("favor_reason", "")).strip()
        # favor_enriched = f"关于{target}，支持者通常认为：{favor}"
        against = str(info.get("against_reason", "")).strip()
        # against_enriched = f"关于{target}，反对者通常认为：{against}"
        if not (background and favor and against):
            logger.warning("target %r has empty stance view(s); falling back to the raw target text", target)
        # views = [target, background_enriched, favor_enriched, against_enriched]
        views = [target, background, favor, against]
        rows.append((target, views))

    logger.info(
        "loaded %d target(s) from %s (types: %s)",
        len(rows),
        template_path,
        dict(
            sorted(
                __import__("collections").Counter(
                    str(info.get("target_type", "")) for info in template.values()
                ).items()
            )
        ),
    )
    return rows


def run_target_knowledge_stage(
    cfg: BuildConfig,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Encode the per-target stance views with the frozen PLM and save them."""
    start = time.time()
    rows = load_target_template(cfg.target_template)
    texts = [text for _, views in rows for text in views]

    flat = encode_unique_sentences(
        texts, tokenizer, model, device, cfg.max_length, cfg.batch_size, cfg.fp16
    )
    knowledge = flat.reshape(len(rows), 4, -1)
    if cfg.normalize:
        norms = np.linalg.norm(knowledge, axis=-1, keepdims=True)
        knowledge = knowledge / np.clip(norms, 1e-9, None)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "target_knowledge.npy"
    with open(npy_path, "wb") as f:
        np.save(f, knowledge.astype(np.float32))

    index = {target: i for i, (target, _) in enumerate(rows)}
    index_path = output_dir / "target_knowledge_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True, ensure_ascii=False)

    logger.info(
        "target knowledge written to %s | shape=%s | %d target(s) in %.2fs",
        npy_path,
        tuple(knowledge.shape),
        len(rows),
        time.time() - start,
    )
    logger.info("target index written to %s", index_path)


def main() -> None:
    args = build_parser().parse_args()
    cfg = BuildConfig(
        data_dir=args.data_dir,
        bert_dir=args.bert_dir,
        output_dir=args.output_dir,
        stage=args.stage,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        normalize=args.normalize,
        fp16=args.fp16,
    )
    logger.info("config:\n%s", cfg.to_json_string())
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.stage in ("dedup", "all"):
        start = time.time()
        files = load_raw_records(cfg.data_dir)
        files, sentence_rows = assign_global_ids(files)
        write_dedup_outputs(files, sentence_rows, output_dir)
        logger.info("dedup stage finished in %.2fs", time.time() - start)

    if cfg.stage not in ("embed", "targets", "all"):
        return

    if not os.path.isdir(cfg.bert_dir):
        raise FileNotFoundError(
            "Please pass a valid --bert_dir (e.g. a local chinese-roberta-wwm-ext folder). "
            f"Got: {cfg.bert_dir!r}"
        )

    device = resolve_device(cfg.device)
    logger.info("using device: %s", device)
    logger.info("loading tokenizer/model from %s ...", cfg.bert_dir)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_dir)
    model = AutoModel.from_pretrained(cfg.bert_dir)
    model.to(device)
    logger.info("model loaded in %.2fs", time.time() - start)

    if cfg.stage in ("embed", "all"):
        csv_path = output_dir / "sentence.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found; run the dedup stage first (or use --stage all)"
            )

        sentence_rows = read_sentence_csv(csv_path)
        sentence_rows = sorted(sentence_rows, key=lambda x: x[0])
        expected_ids = list(range(len(sentence_rows)))
        actual_ids = [gid for gid, _ in sentence_rows]
        if actual_ids != expected_ids:
            raise ValueError(
                f"global_ids in {csv_path} are not a dense 0..N-1 sequence; "
                f"got first={actual_ids[:5]}, last={actual_ids[-5:]}"
            )
        sentences = [s for _, s in sentence_rows]
        logger.info("read %d unique sentences from %s", len(sentences), csv_path)

        start = time.time()
        embeddings = encode_unique_sentences(
            sentences, tokenizer, model, device, cfg.max_length, cfg.batch_size, cfg.fp16
        )
        if cfg.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)
        elapsed = time.time() - start
        logger.info("encoding finished in %.2fs", elapsed)
        write_embed_outputs(embeddings, cfg, len(sentence_rows), elapsed)

    if cfg.stage in ("targets", "all"):
        run_target_knowledge_stage(cfg, tokenizer, model, device)


if __name__ == "__main__":
    main()