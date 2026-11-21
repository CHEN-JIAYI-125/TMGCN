#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Four efficiency numbers on one GPU. No 20-epoch retrain.

Total / Trainable come from the training graph. Time is one real epoch
(train + dev + test eval) times epoch_size. Peak GPU is max_memory_allocated
over that epoch.

    python3.9 scripts/measure_efficiency.py config/config_model5_rd_sem.yaml
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_graph import (  # noqa: E402
    _to_device,
    build_dataset,
    build_model,
    count_params,
    gpu_peak_mb,
    load_config,
    predict_and_evaluate,
    set_seed,
    target_graph_collate_fn,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3.9 scripts/measure_efficiency.py <config.yaml>", flush=True)
        sys.exit(2)
    cfg = load_config(sys.argv[1])
    set_seed(cfg.seed)
    device = torch.device(
        "cuda:%d" % cfg.cuda_index if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.cuda_index)

    train_dataset = build_dataset(cfg, cfg.train_path)
    dev_dataset = build_dataset(cfg, cfg.dev_path)
    test_dataset = build_dataset(cfg, cfg.test_path)
    cfg.input_size = train_dataset.embedding_store.dim
    model = build_model(cfg)
    model.to(device)
    stats = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batchsize,
        shuffle=True,
        collate_fn=target_graph_collate_fn,
        drop_last=False,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.time()
    model.train()
    for batch in loader:
        batch = _to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out["loss"]
        if loss is not None and torch.isfinite(loss):
            loss.backward()
            opt.step()
    predict_and_evaluate(model, dev_dataset, cfg, device, prefix="dev", pred_path=None)
    predict_and_evaluate(model, test_dataset, cfg, device, prefix="test", pred_path=None)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_sec = float(time.time() - t0)
    epochs = max(1, int(cfg.epoch_size))
    train_sec = epoch_sec * epochs
    peak_mb = gpu_peak_mb()
    row = {
        "total_params": stats["total_params"],
        "trainable_params": stats["trainable_params"],
        "total_params_m": round(stats["total_params"] / 1e6, 2),
        "trainable_params_m": round(stats["trainable_params"] / 1e6, 2),
        "epoch_wall_sec": round(epoch_sec, 1),
        "training_time_sec": round(train_sec, 1),
        "training_time_min": round(train_sec / 60.0, 1),
        "epoch_size": epochs,
        "peak_gpu_mem_mb": None if peak_mb is None else round(peak_mb, 1),
        "device": str(device),
        "encoder_frozen_offline": "Qwen3-Embedding-8B ~8.19B",
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    print(
        "TABLE | Total: %.2fM | Trainable: %.2fM | Time: %.1f min (%d x %.1fs) | Peak GPU: %s"
        % (
            row["total_params_m"],
            row["trainable_params_m"],
            row["training_time_min"],
            epochs,
            epoch_sec,
            ("%.1f MB" % peak_mb) if peak_mb is not None else "n/a",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
