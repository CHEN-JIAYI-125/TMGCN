#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在冻结的会话 Embedding 上训练独立 Linear / MLP 分类器。

输入：extract_conversation_embeddings.py 写出的缓存
标签：index.json 里的 label（last_sentence 时 = 官方最后一句立场）
划分：沿用原 train / dev / test 文件，三个 split 分别缓存，不会混用。
不更新 Embedding 模型。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from extract_conversation_embeddings import load_split_cache
from metrics import format_split_line, scoped_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_classifier")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinearClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def labeled_tensors(
    embeddings: np.ndarray,
    index_rows: List[Dict[str, Any]],
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], int]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    types: List[str] = []
    skipped = 0
    if embeddings.shape[0] != len(index_rows):
        raise ValueError("embeddings / index 行数不一致")
    for i, row in enumerate(index_rows):
        lab = int(row["label"])
        if lab < 0 or lab >= num_classes:
            skipped += 1
            continue
        xs.append(embeddings[i])
        ys.append(lab)
        types.append(str(row.get("target_type", "") or ""))
    if not xs:
        raise ValueError("该 split 没有合法标签（0/1/2）")
    x = torch.from_numpy(np.stack(xs, axis=0).astype(np.float32))
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, types, skipped


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    target_types: List[str],
    device: torch.device,
    num_classes: int,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    preds: List[int] = []
    golds: List[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            losses.append(float(criterion(logits, yb).cpu()))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            golds.extend(yb.cpu().tolist())
    metrics = scoped_metrics(golds, preds, target_types, num_classes=num_classes)
    return float(np.mean(losses) if losses else 0.0), metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Linear/MLP on frozen CSD conversation embeddings")
    p.add_argument("--cache_dir", type=str, required=True, help="extract 写出的目录，内含 train/dev/test")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--classifier", type=str, default="mlp", choices=["linear", "mlp"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt", type=str, default="", help="eval_only 时加载的 best.pt")
    return p.parse_args()


def build_model(name: str, dim: int, num_classes: int, hidden: int, dropout: float) -> nn.Module:
    if name == "linear":
        return LinearClassifier(dim, num_classes)
    return MLPClassifier(dim, num_classes, hidden=hidden, dropout=dropout)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    dims = []
    units = []
    models = []
    for split in ("train", "dev", "test"):
        emb, index_rows, config = load_split_cache(cache_dir / split)
        x, y, types, skipped = labeled_tensors(emb, index_rows, args.num_classes)
        splits[split] = {"x": x, "y": y, "types": types, "config": config, "n_raw": emb.shape[0]}
        dims.append(int(emb.shape[1]))
        units.append(config.get("unit"))
        models.append(config.get("model"))
        logger.info(
            "%s: raw_rows=%d labeled=%d skipped=%d dim=%d unit=%s",
            split,
            emb.shape[0],
            x.size(0),
            skipped,
            emb.shape[1],
            config.get("unit"),
        )
        if config.get("dummy"):
            logger.warning("%s cache 是 dummy 向量，不能当作真实实验结果", split)

    if len(set(dims)) != 1:
        raise ValueError("三个 split 的 embedding 维度不一致: %s" % dims)
    if len(set(units)) != 1:
        raise ValueError("三个 split 的 unit 不一致: %s" % units)
    dim = dims[0]
    if units[0] == "prefix":
        logger.info(
            "unit=prefix：每行是因果前缀表示，标签是该 turn 的 all_labels；"
            "官方 CSD 指标仍是会话最后一句，本脚本对缓存中的每一行计分。"
            "若只要官方口径，请用 --unit last_sentence 重新提取。"
        )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    loaders = {}
    for split, pack in splits.items():
        ds = TensorDataset(pack["x"], pack["y"])
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            drop_last=False,
        )

    model = build_model(args.classifier, dim, args.num_classes, args.hidden, args.dropout).to(device)

    if args.eval_only:
        ckpt_path = Path(args.ckpt) if args.ckpt else output_dir / "best.pt"
        blob = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(blob["model_state_dict"])
        loss, metrics = evaluate(model, loaders["test"], splits["test"]["types"], device, args.num_classes)
        print(format_split_line("Final Test", loss, metrics), flush=True)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(running) if running else 0.0)
        dev_loss, dev_metrics = evaluate(
            model, loaders["dev"], splits["dev"]["types"], device, args.num_classes
        )
        print(format_split_line("Epoch %d Dev" % epoch, dev_loss, dev_metrics), flush=True)
        logger.info("epoch %d train_loss=%.4f", epoch, train_loss)
        f1 = float(dev_metrics["mixed_f1_macro"])
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            test_loss, test_metrics = evaluate(
                model, loaders["test"], splits["test"]["types"], device, args.num_classes
            )
            print(format_split_line("Epoch %d Test (best dev)" % epoch, test_loss, test_metrics), flush=True)
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                logger.info("early stop at epoch %d", epoch)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate(
        model, loaders["test"], splits["test"]["types"], device, args.num_classes
    )
    print(format_split_line("Final Test (best epoch %d)" % best_epoch, test_loss, test_metrics), flush=True)

    ckpt = {
        "model_state_dict": best_state,
        "classifier": args.classifier,
        "dim": dim,
        "num_classes": args.num_classes,
        "best_epoch": best_epoch,
        "best_dev_f1_macro": best_f1,
        "test": test_metrics,
        "cache_dir": str(cache_dir),
        "embedding_model": models[0],
        "unit": units[0],
        "seed": args.seed,
    }
    torch.save(ckpt, output_dir / "best.pt")
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_dev_f1_macro": best_f1,
                "test": {k: v for k, v in test_metrics.items() if not str(k).endswith("_per_class")},
                "classifier": args.classifier,
                "unit": units[0],
                "embedding_model": models[0],
                "seed": args.seed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("saved %s", output_dir / "best.pt")


if __name__ == "__main__":
    main()
