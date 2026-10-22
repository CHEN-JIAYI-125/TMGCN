#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: xiongjie <xiongjiezk@163.com>
# @date: 2026/6/8

"""Training entry for the node-level stance tracer.

Pipeline
--------
    raw_with_global_ids/*.json  +  embeddings.npy
        -> TargetGraphDataset (dataset_graph_3 for GraphTracer_4,
                               dataset_graph_4 for GraphTracer_5)
        -> HeterogeneousStanceTracer (model_graph_4) or
           HeterogeneousStanceTracerV5 (model_graph_5, PyG operators)
        -> per-utterance (node-level) stance predictions

Differences from the old dialogue-level entry (model_graph_3)
-------------------------------------------------------------
- Features are frozen offline embeddings; no PLM is loaded at train time.
- A "sample" is one full target graph; every sentence node is labeled and
  predicted (utterance-level evaluation). There is no dialogue-level label.
- Prediction files are therefore written per utterance node, not per
  dialogue. Columns:

      doc_id | target | utterance_id | sentence_idx | speaker_id | true | pred

  `utterance_id` is the global id assigned by the offline stage (row index
  into embeddings.npy); `sentence_idx` is the 0-based turn inside the
  dialogue. This format cannot be the same as the old per-dialogue CSV
  because the prediction unit changed from "one dialogue" to "every
  sentence node".

No HF Trainer: target graphs have very different sizes, so a plain
DataLoader + manual loop keeps batching, gradient accumulation and
evaluation simple and predictable.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from yaml import safe_load

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from datasets import TargetGraphDataset, target_graph_collate_fn
from models import HeterogeneousStanceTracerV5


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# 终端和 nohup 日志都保留 INFO，方便 tail -f 看每一轮数字


def _pct(x: float) -> str:
    return f"{float(x) * 100.0:.2f}"


def format_split_line(title: str, loss: float, metrics: Dict[str, Any]) -> str:
    """和 ZS-CSD 终端一致：Epoch N Dev | Loss | Macro F1 | Favor | Against | Neutral。"""
    return (
        f"{title} | Loss: {float(loss):.2f} | Macro F1: {_pct(metrics['mixed_f1_macro'])} | "
        f"Favor: {_pct(metrics['mixed_f1_favor'])} | Against: {_pct(metrics['mixed_f1_against'])} | "
        f"Neutral: {_pct(metrics['mixed_f1_neutral'])}"
    )


def print_split_line(title: str, loss: float, metrics: Dict[str, Any]) -> None:
    line = format_split_line(title, loss, metrics)
    print(line, flush=True)
    logger.info(line)


def _iter_with_progress(iterable, total=None, desc=""):
    # nohup / 重定向时不用 tqdm，避免 \r 进度条把 tail 刷乱
    if tqdm is None or not sys.stdout.isatty():
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=True, dynamic_ncols=True)


@dataclass
class GraphConfig:
    # data (augmented raw files produced by build_embeddings.py --stage dedup)
    train_path: str = field(default="offline/raw_with_global_ids/train_data.json")
    dev_path: str = field(default="offline/raw_with_global_ids/dev_data.json")
    test_path: str = field(default="offline/raw_with_global_ids/test_data.json")
    embedding_npy: str = field(default="offline/embeddings.npy")
    # frozen per-target stance-view prior ([T, 4, D], --stage targets);
    # empty string disables target-side grounding features
    target_knowledge_npy: str = field(default="offline/target_knowledge.npy")

    # model
    hidden_size: int = field(default=768)
    input_size: int = field(default=0)  # 0: infer frozen feature width
    num_classes: int = field(default=3)
    dropout: float = field(default=0.1)
    input_dropout: float = field(default=0.0)
    reply_temperature: float = field(default=1.0)
    use_gumbel: bool = field(default=True)
    num_layers: int = field(default=2)
    # v5 (PyG tracer) options
    rgcn_num_bases: int = field(default=0)  # 0 -> no basis decomposition
    hypergraph_attention: bool = field(default=True)
    # v5 representation-level contrastive regularization (cl_lambda <= 0 disables)
    cl_lambda: float = field(default=0.0)
    # "supcon" (paper L_CL: label-aware weighted SupCon) | "augment" (legacy
    # self-supervised InfoNCE between two corrupted views of h)
    cl_mode: str = field(default="supcon")
    # SupCon positive-pair weights: same stance + same target vs same stance
    # only (paper: higher weight when both are shared)
    cl_pos_weight_same_target: float = field(default=1.0)
    cl_pos_weight_diff_target: float = field(default=0.5)
    cl_temperature: float = field(default=0.2)
    cl_feat_mask_rate: float = field(default=0.2)
    cl_noise_scale: float = field(default=0.1)
    cl_proj_dim: int = field(default=128)
    # v5 target-side stance semantic grounding (needs target_knowledge_npy)
    use_grounding: bool = field(default=False)
    # 通道开关：0 关 / 1 开（由 yaml 控制，不在 __init__.py 里选）
    use_speaker: int = field(default=1)
    use_dialogue: int = field(default=1)
    use_temporal: int = field(default=1)
    use_reply: int = field(default=1)

    # graph construction
    bidirectional_temporal: bool = field(default=False)
    max_nodes_per_target: int = field(default=0)  # 0 -> full graph (no sampling)

    # training
    epoch_size: int = field(default=20)
    batchsize: int = field(default=1)  # target graphs per step
    patience: int = field(default=0)  # 0 = 不 early stop，跑满 epoch_size
    max_grad_norm: float = field(default=1.0)
    warmup_proportion: float = field(default=0.1)
    warmup_steps: int = field(default=0)
    gradient_accumulation_steps: int = field(default=1)
    adam_epsilon: float = field(default=1e-8)
    lr: float = field(default=1e-4)
    weight_decay: float = field(default=1e-5)
    logging_steps: int = field(default=10)

    # runtime
    cuda_index: int = field(default=0)
    seed: int = field(default=1234)
    debug: bool = field(default=False)
    time: str = field(default="")
    model_type: str = field(default="GraphTracer_4")
    result_root: str = field(default="./result")

    # evaluation: score only the last sentence of each doc_id (official CSD
    # setting). Full per-utterance predictions are still written to CSV.
    metric_on_last_sentence: bool = field(default=True)

    # Reply Distillation 方法开关：0 关 / 1 开（需先在 models/datasets 的 __init__.py 换成 rd）
    reply_distillation: int = field(default=0)
    use_reply_distillation: int = field(default=0)
    teacher_path: str = field(default="data/reply_teacher/")
    reply_kd_lambda: float = field(default=0.5)
    reply_threshold: float = field(default=0.05)
    use_reply_kd: int = field(default=0)
    use_soft_reply_graph: int = field(default=0)
    reply_graph_source: str = field(default="student")
    reply_ablation: str = field(default="")

    def to_json_string(self, *args, **kwargs):
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False)


def _reply_on(cfg: GraphConfig) -> bool:
    return bool(cfg.reply_distillation or cfg.use_reply_distillation)


def apply_reply_ablation(cfg: GraphConfig) -> GraphConfig:
    """reply_ablation=A..E 只改 0/1 方法开关，不切换 __init__.py 里的模型文件。"""
    mode = str(cfg.reply_ablation or "").strip().upper()
    if not mode:
        return cfg
    mapping = {
        "A": dict(distill=0, kd=0, soft=0, src="candidate"),
        "B": dict(distill=1, kd=0, soft=1, src="student"),
        "C": dict(distill=1, kd=1, soft=0, src="candidate"),
        "D": dict(distill=1, kd=0, soft=1, src="teacher"),
        "E": dict(distill=1, kd=1, soft=1, src="student"),
    }
    if mode not in mapping:
        raise ValueError("reply_ablation 必须是 A/B/C/D/E 或空，收到 %r" % cfg.reply_ablation)
    spec = mapping[mode]
    cfg.reply_distillation = spec["distill"]
    cfg.use_reply_distillation = spec["distill"]
    cfg.use_reply_kd = spec["kd"]
    cfg.use_soft_reply_graph = spec["soft"]
    cfg.reply_graph_source = spec["src"]
    logger.info(
        "reply_ablation=%s -> distillation=%d kd=%d soft_graph=%d source=%s",
        mode, spec["distill"], spec["kd"], spec["soft"], spec["src"],
    )
    return cfg


def load_config(config_path: str) -> GraphConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = safe_load(f) or {}
    known = {name for name in GraphConfig.__dataclass_fields__}
    obj = GraphConfig(**{k: v for k, v in cfg.items() if k in known})
    return apply_reply_ablation(obj)


def get_collate_fn(cfg: GraphConfig):
    return target_graph_collate_fn


def build_model(cfg: GraphConfig):
    # 用哪个模型由 models/__init__.py 决定
    return HeterogeneousStanceTracerV5(cfg)


def build_dataset(cfg: GraphConfig, split_path: str) -> Any:
    import inspect

    max_nodes = None if cfg.max_nodes_per_target <= 0 else cfg.max_nodes_per_target
    knowledge_npy = (cfg.target_knowledge_npy or "").strip()
    if knowledge_npy and not os.path.isfile(knowledge_npy):
        knowledge_npy = ""
        logger.warning(
            "target_knowledge_npy %s not found; target-side grounding prior disabled",
            cfg.target_knowledge_npy,
        )
    kwargs = dict(
        raw_dir=split_path,
        embedding_npy=cfg.embedding_npy,
        max_nodes_per_target=max_nodes,
        bidirectional_temporal=cfg.bidirectional_temporal,
        seed=cfg.seed,
    )
    sig = inspect.signature(TargetGraphDataset.__init__)
    if "target_knowledge_npy" in sig.parameters:
        kwargs["target_knowledge_npy"] = knowledge_npy or None
    if "teacher_path" in sig.parameters:
        kwargs["teacher_path"] = cfg.teacher_path
        kwargs["reply_distillation"] = _reply_on(cfg)
    ds = TargetGraphDataset(**kwargs)
    if cfg.debug:
        keep = min(len(ds), 40)
        ds.target_ids = ds.target_ids[:keep]
        ds.target_texts = ds.target_texts[:keep]
        if hasattr(ds, "target_types"):
            ds.target_types = ds.target_types[:keep]
        ds.graphs = ds.graphs[:keep]
        if hasattr(ds, "reply_teacher_dist"):
            ds.reply_teacher_dist = ds.reply_teacher_dist[:keep]
            ds.reply_teacher_mask = ds.reply_teacher_mask[:keep]
        logger.info("[debug] trimmed dataset to %d target graphs", keep)
    return ds


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_optimizer(model: torch.nn.Module, cfg: GraphConfig) -> torch.optim.Optimizer:
    no_decay = {"bias", "LayerNorm.weight", "norm.weight", "norm.bias"}
    decay, nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            nodecay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": cfg.weight_decay})
    if nodecay:
        groups.append({"params": nodecay, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, eps=cfg.adam_epsilon)
    logger.info("[optimizer] lr=%.2e weight_decay=%.2e eps=%.2e", cfg.lr, cfg.weight_decay, cfg.adam_epsilon)
    return optimizer


def build_scheduler(optimizer, cfg: GraphConfig, total_steps: int):
    warmup_steps = cfg.warmup_steps if cfg.warmup_steps > 0 else int(total_steps * cfg.warmup_proportion)
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# def build_scheduler(optimizer, cfg: GraphConfig, total_steps: int):
#     # 完全忽略预热配置，直接使用余弦退火
#     # eta_min 是最终学习率缩放因子，建议设为 0 或 1e-6 倍初始 lr
#     return torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, 
#         T_max=total_steps, 
#         eta_min=0  # 或 0.01 * cfg.lr，根据需求
#     )


def _class_f1s(y_true, y_pred, num_classes: int) -> List[float]:
    per_class = f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0.0)
    return [float(v) for v in per_class]


def utterance_metrics(y_true, y_pred, num_classes: int) -> Dict[str, float]:
    """Utterance-level metrics over all sentence nodes (labels >= 0 only)."""
    f1s = _class_f1s(y_true, y_pred, num_classes)
    counts = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=num_classes)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)),
        "f1_avg": float(np.mean(f1s)),
        "f1_per_class": f1s,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "majority_acc": float(counts.max() / max(1, len(y_true))),
        "num_nodes": int(len(y_true)),
    }


# CSD target-type scopes: "mixed" = all targets, "c" = claim targets,
# "n" = noun-phrase targets (same口径 as the legacy main_12 evaluator).
TARGET_TYPE_SCOPES = ("mixed", "c", "n")


def scoped_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_types: List[str],
    num_classes: int,
) -> Dict[str, Any]:
    """Metrics overall plus per target type (mixed / claim / noun-phrase).

    `target_types` is parallel to `y_true`/`y_pred` and holds the target's
    type for each scored row. For every scope `s` in {mixed, c, n} the
    output carries keys `s_f1_macro`, `s_f1_favor`, `s_f1_against`,
    `s_f1_neutral`, `s_f1_avg`, `s_f1_per_class` and `s_accuracy`, plus
    `s_num` (number of scored rows in that scope). Empty scopes report 0.0
    everywhere, matching the legacy main_12 behavior. Class order follows
    the CSD convention used by main_12: 0 = favor, 1 = against, 2 = neutral.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    types_arr = np.asarray([str(t) for t in target_types])

    scopes = {
        "mixed": np.ones(types_arr.shape, dtype=bool),
        "c": types_arr == "c",
        "n": types_arr == "n",
    }

    metrics: Dict[str, Any] = {}
    for scope_name, mask in scopes.items():
        n_scored = int(mask.sum())
        if n_scored == 0:
            metrics.update(
                {
                    f"{scope_name}_f1_macro": 0.0,
                    f"{scope_name}_f1_favor": 0.0,
                    f"{scope_name}_f1_against": 0.0,
                    f"{scope_name}_f1_neutral": 0.0,
                    f"{scope_name}_f1_avg": 0.0,
                    f"{scope_name}_f1_per_class": [0.0] * num_classes,
                    f"{scope_name}_accuracy": 0.0,
                    f"{scope_name}_num": 0,
                }
            )
            continue

        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]
        f1_macro = float(f1_score(yt, yp, average="macro", zero_division=0.0))
        f1_per_class = _class_f1s(yt, yp, num_classes)
        # CSD label convention: 0 = favor, 1 = against, 2 = neutral
        favor = f1_per_class[0] if num_classes > 0 else 0.0
        against = f1_per_class[1] if num_classes > 1 else 0.0
        neutral = f1_per_class[2] if num_classes > 2 else 0.0
        f1_avg = (favor + neutral + against) / 3.0
        metrics.update(
            {
                f"{scope_name}_f1_macro": f1_macro,
                f"{scope_name}_f1_favor": float(favor),
                f"{scope_name}_f1_against": float(against),
                f"{scope_name}_f1_neutral": float(neutral),
                f"{scope_name}_f1_avg": float(f1_avg),
                f"{scope_name}_f1_per_class": f1_per_class,
                f"{scope_name}_accuracy": float(accuracy_score(yt, yp)),
                f"{scope_name}_num": n_scored,
            }
        )
    return metrics


def _log_scoped_metrics(prefix: str, metrics: Dict[str, Any]) -> None:
    logger.info(
        "%s metrics mixed | f1_macro=%.6f | f1_avg=%.6f | accuracy=%.6f | scored=%d",
        prefix.upper(),
        metrics["mixed_f1_macro"],
        metrics["mixed_f1_avg"],
        metrics["mixed_accuracy"],
        metrics["mixed_num"],
    )
    for scope in ("c", "n"):
        logger.info(
            "%s metrics %s | f1_macro=%.6f | f1_avg=%.6f | accuracy=%.6f | "
            "favor=%.4f | against=%.4f | neutral=%.4f | scored=%d",
            prefix.upper(),
            "claim(c)" if scope == "c" else "noun-phrase(n)",
            metrics[f"{scope}_f1_macro"],
            metrics[f"{scope}_f1_avg"],
            metrics[f"{scope}_accuracy"],
            metrics[f"{scope}_f1_favor"],
            metrics[f"{scope}_f1_against"],
            metrics[f"{scope}_f1_neutral"],
            metrics[f"{scope}_num"],
        )


def predict_and_evaluate(
    model: torch.nn.Module,
    dataset: Any,  # dataset_graph_3.TargetGraphDataset or dataset_graph_4.TargetGraphDataset
    cfg: GraphConfig,
    device: torch.device,
    prefix: str,
    pred_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one full-graph forward per target and evaluate sentence nodes.

    One row per **unique sentence node** is written to `pred_path`.

    Metrics use the per-doc final-sentence mapping kept on the graph
    (`doc_final_*`), because a doc's last sentence can be a shared node whose
    first occurrence belongs to another doc. When
    `cfg.metric_on_last_sentence` is False, every labeled node is scored.
    """
    model.eval()
    node_rows: List[Dict[str, Any]] = []
    doc_rows: List[Dict[str, Any]] = []
    eval_loss_sum = 0.0
    eval_loss_n = 0

    with torch.no_grad():
        for idx in _iter_with_progress(range(len(dataset)), total=len(dataset), desc=""):
            item = dataset[idx]
            batch = _to_device(get_collate_fn(cfg)([item]), device)
            output = model(**batch)
            loss = output.get("loss")
            if loss is not None and torch.is_tensor(loss) and torch.isfinite(loss):
                eval_loss_sum += float(loss.detach().cpu())
                eval_loss_n += 1
            logits = output["logits"]
            y = batch["y"].cpu().tolist()
            preds = logits.argmax(dim=-1).cpu().tolist()
            target_text = dataset.target_texts[idx]
            target_type = getattr(dataset, "target_types", [""] * len(dataset))[idx]

            doc_ids = batch["dialogue_ids"].cpu().tolist()
            utterance_ids = batch["utterance_ids"].cpu().tolist()
            turn_index = batch["turn_index"].cpu().tolist()
            speaker_ids = batch["speaker_ids"].cpu().tolist()

            for i in range(logits.size(0)):
                node_rows.append(
                    {
                        "doc_id": doc_ids[i],
                        "target": target_text,
                        "target_type": target_type,
                        "utterance_id": utterance_ids[i],
                        "sentence_idx": turn_index[i],
                        "speaker_id": speaker_ids[i],
                        "true": y[i],
                        "pred": preds[i],
                    }
                )

            # doc-level scoring: the official CSD unit is the final sentence
            final_doc_ids = batch["doc_final_doc_ids"].cpu().tolist()
            final_nodes = batch["doc_final_node_index"].cpu().tolist()
            final_turns = batch["doc_final_turn_index"].cpu().tolist()
            final_true = batch["doc_final_true"].cpu().tolist()
            for doc, node, turn, true in zip(final_doc_ids, final_nodes, final_turns, final_true):
                doc_rows.append(
                    {
                        "doc_id": doc,
                        "target": target_text,
                        "target_type": target_type,
                        "utterance_id": utterance_ids[node],
                        "sentence_idx": turn,
                        "speaker_id": speaker_ids[node],
                        "true": true,
                        "pred": preds[node],
                    }
                )

    if pred_path is not None and node_rows:
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        # utf-8-sig adds a BOM so Excel on Windows renders Chinese correctly;
        # the file itself is still UTF-8 text.
        pd.DataFrame(node_rows).to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info(
            "%s node predictions written to %s (%d unique sentences)",
            prefix.upper(),
            pred_path,
            len(node_rows),
        )

    if cfg.metric_on_last_sentence:
        metric_rows = doc_rows
        logger.debug(
            "%s scoring %d doc_id(s) via their final-sentence mapping",
            prefix.upper(),
            len(metric_rows),
        )
    else:
        metric_rows = node_rows

    y_true_list = [r["true"] for r in metric_rows if r["true"] >= 0]
    y_pred_list = [r["pred"] for r in metric_rows if r["true"] >= 0]
    target_types_list = [str(r.get("target_type", "")) for r in metric_rows if r["true"] >= 0]
    metrics = utterance_metrics(y_true_list, y_pred_list, cfg.num_classes)
    metrics["prefix"] = prefix
    metrics["eval_loss"] = eval_loss_sum / max(1, eval_loss_n)
    metrics.update(scoped_metrics(y_true_list, y_pred_list, target_types_list, cfg.num_classes))
    _log_scoped_metrics(prefix, metrics)
    return metrics


def main():
    current_dir = Path(__file__).resolve().parent
    default_candidates = [
        current_dir / "config" / "config_model5.yaml",
        current_dir / "config" / "config_model4.yaml",
        current_dir / "config" / "config.yaml",
    ]
    if len(sys.argv) > 1 and not str(sys.argv[1]).startswith("-"):
        config_path = Path(sys.argv[1])
    else:
        config_path = next((p for p in default_candidates if p.exists()), default_candidates[0])
    cfg = load_config(str(config_path))
    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.cuda_index}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.cuda_index)

    timestamp = cfg.time or time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path(cfg.result_root) / cfg.model_type / timestamp
    log_dir = base_dir / "log"
    pred_dir = base_dir / "pred"
    save_dir = base_dir / "save"
    log_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / f"{cfg.seed}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    logger.info("config file: %s", config_path)
    logger.info("config:\n%s", cfg.to_json_string())

    train_dataset = build_dataset(cfg, cfg.train_path)
    dev_dataset = build_dataset(cfg, cfg.dev_path)
    test_dataset = build_dataset(cfg, cfg.test_path)
    logger.info(
        "target graphs | train=%d dev=%d test=%d",
        len(train_dataset),
        len(dev_dataset),
        len(test_dataset),
    )

    feature_dims = {ds.embedding_store.dim for ds in (train_dataset, dev_dataset, test_dataset)}
    if len(feature_dims) != 1:
        raise ValueError(f"Split embedding dimensions differ: {feature_dims}")
    feature_dim = feature_dims.pop()
    if cfg.input_size not in (0, feature_dim):
        raise ValueError(f"input_size={cfg.input_size} but embeddings have width {feature_dim}")
    cfg.input_size = feature_dim
    print(
        f"embeddings: {cfg.embedding_npy} | dim={feature_dim}",
        flush=True,
    )
    model = build_model(cfg)
    model.to(device)
    logger.info(
        "model params: %.2fM",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batchsize,
        shuffle=True,
        collate_fn=get_collate_fn(cfg),
        drop_last=False,
    )
    steps_per_epoch = max(1, len(train_loader) // cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.epoch_size
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    global_step = 0
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    tr_loss = 0.0
    tr_steps = 0

    for epoch in range(1, cfg.epoch_size + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        train_iter = _iter_with_progress(train_loader, total=len(train_loader), desc="")
        for step, batch in enumerate(train_iter, start=1):
            batch = _to_device(batch, device)
            output = model(**batch)
            loss = output["loss"]
            if not torch.isfinite(loss):
                logger.warning("[train] non-finite loss=%s at step %d; skipping step", float(loss), global_step)
                continue
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            epoch_loss += float(loss.detach().cpu())
            tr_loss += float(loss.detach().cpu())
            tr_steps += 1

            if step % cfg.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    avg = tr_loss / max(1, tr_steps)
                    lr_now = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "[train] step=%d | epoch=%.4f | loss=%.6f | grad_norm=%.4f | lr=%.2e",
                        global_step,
                        epoch + step / len(train_loader),
                        avg,
                        float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))),
                        lr_now,
                    )
                    tr_loss = 0.0
                    tr_steps = 0

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} Train | Loss: {avg_epoch_loss:.2f}", flush=True)
        logger.info("epoch %d finished | avg_loss=%.6f", epoch, avg_epoch_loss)

        dev_metrics = predict_and_evaluate(
            model, dev_dataset, cfg, device, prefix="dev",
            pred_path=str(pred_dir / f"epoch_{epoch}_dev.csv"),
        )
        print_split_line(f"Epoch {epoch} Dev", dev_metrics["eval_loss"], dev_metrics)

        test_metrics = predict_and_evaluate(
            model, test_dataset, cfg, device, prefix="test",
            pred_path=str(pred_dir / f"epoch_{epoch}_test.csv"),
        )
        print_split_line(f"Epoch {epoch} Test", test_metrics["eval_loss"], test_metrics)

        dev_f1 = dev_metrics["mixed_f1_macro"]
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            logger.info("new best dev f1_macro=%.6f (saved, epoch=%d)", dev_f1, epoch)
        else:
            bad_epochs += 1
            logger.info("dev f1_macro=%.6f (no improvement, bad_epochs=%d)", dev_f1, bad_epochs)

        # patience<=0：按 epoch_size 跑满，不 early stop

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    save_path = save_dir / "best_model.pt"
    torch.save({"config": asdict(cfg), "model_state_dict": best_state}, save_path)
    logger.info("best model saved to %s (dev f1_macro=%.6f)", save_path, best_f1)

    # Restore the best dev model and run final dev/test predictions.
    model.load_state_dict(best_state)
    model.to(device)
    dev_metrics = predict_and_evaluate(
        model, dev_dataset, cfg, device, prefix="dev",
        pred_path=str(pred_dir / "final_dev.csv"),
    )
    test_metrics = predict_and_evaluate(
        model, test_dataset, cfg, device, prefix="test",
        pred_path=str(pred_dir / "final_test.csv"),
    )
    print_split_line(
        f"Final Test (best epoch {best_epoch or 1})",
        test_metrics["eval_loss"],
        test_metrics,
    )

    summary = {
        "best_dev_f1_macro": best_f1,
        "best_epoch": best_epoch,
        "dev": dev_metrics,
        "test": test_metrics,
        "config": asdict(cfg),
    }
    with open(base_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
    logger.info("metrics summary written to %s", base_dir / "metrics_summary.json")


if __name__ == "__main__":
    main()
