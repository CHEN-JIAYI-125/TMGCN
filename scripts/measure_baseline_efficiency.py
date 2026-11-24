#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure RoBERTa / Branch-BERT / GLAN / SITPCL on TMGCN ZS-CSD JSON.

Same protocol as scripts/measure_efficiency.py: one train epoch (train+dev)
then one test pass. Training Time = epoch_wall * 20. Peak GPU is
max_memory_allocated. Inference is the test pass only.

Uses Chinese-RoBERTa-wwm-ext for all four, matching Ding et al. (ZS-CSD).
Default batch size 16 is the ZS-CSD paper setting for every PLM baseline
(Ding et al., ACL 2025 Findings, §5.2). Sequence cap is BERT's 512:
Branch-BERT concatenates the branch and truncates at 512 (Li et al.);
GLAN concatenates the conversation; SITPCL encodes each utterance-target
pair so the whole dialogue is not packed into one 512 window. Padding is
dynamic (pad to the longest item in the batch, not to 512).

This run is for efficiency, not Table-1 F1.

    python3.9 scripts/measure_baseline_efficiency.py --model all --cuda-index 0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _ensure_torch_distributed_tensor():
    """transformers>=4.49 imports torch.distributed.tensor (PyTorch 2.5+).

    Do not upgrade torch on this server. Alias the 2.1–2.4 private module, or
    install a dummy so BertModel can load on a single GPU.
    """
    try:
        import torch.distributed.tensor  # noqa: F401
        return
    except Exception:
        pass
    import sys
    import types

    try:
        import torch.distributed._tensor as _dt

        sys.modules["torch.distributed.tensor"] = _dt
        return
    except Exception:
        pass
    dummy = types.ModuleType("torch.distributed.tensor")

    class _Place:
        def __init__(self, *args, **kwargs):
            pass

    dummy.DTensor = type("DTensor", (), {})
    dummy.DeviceMesh = type("DeviceMesh", (), {})
    dummy.Replicate = _Place
    dummy.Shard = _Place
    dummy.Partial = _Place
    dummy.distribute_tensor = lambda *a, **k: a[0] if a else None
    dummy.distribute_module = lambda *a, **k: a[0] if a else None
    sys.modules["torch.distributed.tensor"] = dummy


_ensure_torch_distributed_tensor()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = "/home/chenjiayi/data/csd/raw_with_all_labels_v3/raw_with_global_ids/train_data.json"
DEFAULT_DEV = "/home/chenjiayi/data/csd/raw_with_all_labels_v3/raw_with_global_ids/dev_data.json"
DEFAULT_TEST = "/home/chenjiayi/data/csd/raw_with_all_labels_v3/raw_with_global_ids/test_data.json"
BERT_CANDIDATES = (
    "/home/chenjiayi/models/chinese-roberta-wwm-ext",
    "/home/chenjiayi/models/hfl-chinese-roberta-wwm-ext",
    "hfl/chinese-roberta-wwm-ext",
)
PAPER_BATCH_SIZE = 16
PAPER_MAX_LENGTH = {
    "roberta": 512,
    "branch_bert": 512,
    "glan": 512,
    "sitpcl": 512,
}


def paper_max_length(name: str, override: int) -> int:
    if override and override > 0:
        return int(override)
    return PAPER_MAX_LENGTH[name]


def load_docs(path: str) -> List[Dict[str, Any]]:
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for rec in recs:
        sents = [str(s) for s in (rec.get("sentences") or [])]
        if not sents:
            continue
        lab = int(rec.get("label", -1))
        if lab < 0:
            continue
        speakers = rec.get("speakers") or list(range(len(sents)))
        speakers = [int(s) for s in speakers[: len(sents)]]
        while len(speakers) < len(sents):
            speakers.append(speakers[-1] if speakers else 0)
        out.append(
            {
                "sentences": sents[-16:],
                "speakers": speakers[-16:],
                "target": str(rec.get("target", "") or ""),
                "label": lab,
            }
        )
    return out


class DocDS(Dataset):
    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        return self.docs[i]


def count_params(model: nn.Module):
    total = int(sum(p.numel() for p in model.parameters()))
    train = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total, train


def gpu_peak_mb():
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)


class RobertaCls(nn.Module):
    def __init__(self, bert, hidden, n_cls):
        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.fc(h)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return loss, logits


class BranchBERT(nn.Module):
    """BERT + TextCNN (filters 2/3/4, 32 maps) as in Li et al. Branch-BERT."""

    def __init__(self, bert, hidden, n_cls):
        super().__init__()
        self.bert = bert
        self.convs = nn.ModuleList([nn.Conv1d(hidden, 32, k) for k in (2, 3, 4)])
        self.fc = nn.Linear(32 * 3, n_cls)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = h.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            y = F.relu(conv(x))
            pooled.append(y.max(dim=-1).values)
        logits = self.fc(torch.cat(pooled, dim=-1))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return loss, logits


class GraphConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        hidden = x @ self.weight
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adj @ hidden / deg + self.bias


class GLAN(nn.Module):
    """BERT + global / local CNN / 2-layer GCN + target attention (Niu et al.)."""

    def __init__(self, bert, hidden, n_cls):
        super().__init__()
        self.bert = bert
        self.gc1 = GraphConv(hidden)
        self.gc2 = GraphConv(hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.fc = nn.Linear(hidden * 3, n_cls)

    def _utterance_vecs(self, h, spans):
        vecs = []
        for s, e in spans:
            e = max(e, s + 1)
            vecs.append(h[s:e].mean(dim=0))
        return torch.stack(vecs, dim=0)

    def forward_one(self, h, spans, tvec):
        sent = self._utterance_vecs(h, spans)
        n = sent.size(0)
        adj = torch.eye(n, device=sent.device) + torch.diag(torch.ones(n - 1, device=sent.device), 1)
        g = torch.tanh(self.gc2(torch.tanh(self.gc1(sent, adj)), adj))
        loc = torch.tanh(self.conv1(sent.transpose(0, 1)))
        loc = torch.tanh(self.conv2(loc)).transpose(0, 1)
        w = torch.softmax(sent[-1:] @ sent.t(), dim=-1)
        glob = w @ sent
        tw = torch.softmax(torch.stack([g[-1], loc[-1], glob.squeeze(0)]) @ tvec, dim=0)
        parts = torch.stack([g[-1], loc[-1], glob.squeeze(0)]) * tw.unsqueeze(-1)
        return self.fc(parts.reshape(-1))

    def forward(self, input_ids, attention_mask, target_ids, target_mask, spans, labels=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        t = self.bert(input_ids=target_ids, attention_mask=target_mask).pooler_output
        logits = torch.stack(
            [self.forward_one(h[i], spans[i], t[i]) for i in range(h.size(0))], dim=0
        )
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return loss, logits


class SITPCL(nn.Module):
    """RoBERTa + GRU + speaker interaction, Ding et al. SITPCL (CE only)."""

    def __init__(self, bert, hidden, n_cls, gru_layers=1):
        super().__init__()
        self.bert = bert
        self.gru = nn.GRU(hidden, hidden, num_layers=gru_layers, batch_first=True)
        self.linear_intra = nn.Linear(hidden * 2, hidden)
        self.q_intra = nn.Linear(hidden, hidden)
        self.k_intra = nn.Linear(hidden, hidden)
        self.v_intra = nn.Linear(hidden, hidden)
        self.linear_inter = nn.Linear(hidden, hidden)
        self.q_inter = nn.Linear(hidden, hidden)
        self.k_inter = nn.Linear(hidden, hidden)
        self.v_inter = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, n_cls)
        self.scale = hidden ** -0.5

    def _attn(self, q, k, v, qlin, klin, vlin):
        if q.dim() == 1:
            q = q.unsqueeze(0)
        scores = (qlin(q) @ klin(k).t()) * self.scale
        w = torch.softmax(scores, dim=-1)
        return (w @ vlin(v)).squeeze(0)

    def sse(self, utt, speakers):
        v_lst = []
        last = {}
        for i, spk in enumerate(speakers):
            if spk not in last:
                v_lst.append(utt[i])
            else:
                prev = last[spk]
                q_in = self.linear_intra(torch.cat([v_lst[prev], utt[i]], dim=-1))
                ctx = utt[: i + 1]
                v_in = self._attn(q_in, ctx, ctx, self.q_intra, self.k_intra, self.v_intra)
                k = torch.stack(v_lst[prev:i]) if i > prev else utt[i : i + 1]
                q_out = self.linear_inter(utt[i])
                v_out = self._attn(q_out, k, k, self.q_inter, self.k_inter, self.v_inter)
                v_lst.append(v_in + v_out)
            last[spk] = i
        return torch.stack(v_lst)

    def forward(self, input_ids, attention_mask, dia_idx, speakers, labels=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        stance = []
        for b, (st, ed) in enumerate(dia_idx):
            seq = pooled[st:ed]
            o, _ = self.gru(seq.unsqueeze(0))
            v = self.sse(o.squeeze(0), speakers[b])
            stance.append(v[-1])
        logits = self.fc(torch.stack(stance, dim=0))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return loss, logits


def make_collate(name, tok, max_len, device):
    def _enc(texts, texts_pair=None):
        kwargs = dict(
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        if texts_pair is None:
            return tok(texts, **kwargs)
        return tok(texts, texts_pair, **kwargs)

    def collate_pair(batch):
        joined = ["[SEP]".join(b["sentences"]) for b in batch]
        enc = _enc(joined, [b["target"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "labels": labels.to(device),
        }

    def collate_glan(batch):
        convs = [tok.sep_token.join(b["sentences"]) for b in batch]
        targets = [b["target"] for b in batch]
        enc = _enc(convs)
        tenc = _enc(targets)
        real_spans = []
        for i, b in enumerate(batch):
            n = max(1, len(b["sentences"]))
            L = int((enc["attention_mask"][i] > 0).sum())
            step = max(1, (max(L - 2, 1)) // n)
            cuts = [1 + min(k * step, max(L - 2, 1)) for k in range(n)]
            cuts.append(max(L - 1, 2))
            real_spans.append(
                [(cuts[k], max(cuts[k] + 1, cuts[k + 1])) for k in range(n)]
            )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "target_ids": tenc["input_ids"].to(device),
            "target_mask": tenc["attention_mask"].to(device),
            "spans": real_spans,
            "labels": labels.to(device),
        }

    def collate_sitpcl(batch):
        utt_texts = []
        dia_idx = []
        speakers = []
        st = 0
        for b in batch:
            dia_idx.append((st, st + len(b["sentences"])))
            speakers.append(b["speakers"])
            for sen, spk in zip(b["sentences"], b["speakers"]):
                utt_texts.append(
                    "在话语“%s”中，用户%s对%s的立场" % (sen, spk, b["target"])
                )
            st += len(b["sentences"])
        enc = _enc(utt_texts)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "dia_idx": dia_idx,
            "speakers": speakers,
            "labels": labels.to(device),
        }

    if name in ("roberta", "branch_bert"):
        return collate_pair
    if name == "glan":
        return collate_glan
    return collate_sitpcl


def run_split(model, loader, train, device):
    if train:
        model.train()
    else:
        model.eval()
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            labels = batch["labels"]
            n += int(labels.size(0))
            kwargs = {k: v for k, v in batch.items() if k != "labels"}
            if train:
                model.zero_grad(set_to_none=True)
            loss, _ = model(**kwargs, labels=labels)
            if train and loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model._opt.step()
    return n


def resolve_bert(path: str) -> str:
    if path:
        local = Path(path).expanduser()
        if local.is_dir() and (
            (local / "config.json").is_file() or (local / "tokenizer_config.json").is_file()
        ):
            return str(local)
        # Hugging Face hub id, e.g. hfl/chinese-roberta-wwm-ext
        posix = path.replace("\\", "/")
        if "/" in posix and not posix.startswith("/") and not local.exists():
            return path
        raise SystemExit(
            "bert-dir is not a local model folder: %s\n"
            "This machine has no chinese-roberta-wwm-ext under ~/models.\n"
            "Use the hub id (needs HF_ENDPOINT=https://hf-mirror.com):\n"
            "  python3.9 scripts/measure_baseline_efficiency.py --model all "
            "--bert-dir hfl/chinese-roberta-wwm-ext --cuda-index 0\n"
            "Or download:\n"
            "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download "
            "hfl/chinese-roberta-wwm-ext --local-dir /home/chenjiayi/models/chinese-roberta-wwm-ext"
            % path
        )
    for c in BERT_CANDIDATES:
        local = Path(c)
        if local.is_dir() and (local / "config.json").is_file():
            return str(local)
    return BERT_CANDIDATES[-1]


def measure_one(name, args, device, tok, bert_name):
    from transformers import AutoModel

    max_len = paper_max_length(name, args.max_length)
    bs = args.batch_size
    print(
        "=== %s === batch_size=%d max_length=%d padding=dynamic" % (name, bs, max_len),
        flush=True,
    )
    bert = AutoModel.from_pretrained(bert_name)
    hidden = int(bert.config.hidden_size)
    if name == "roberta":
        model = RobertaCls(bert, hidden, 3)
    elif name == "branch_bert":
        model = BranchBERT(bert, hidden, 3)
    elif name == "glan":
        model = GLAN(bert, hidden, 3)
    else:
        model = SITPCL(bert, hidden, 3)
    model.to(device)
    total, trainable = count_params(model)
    model._opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    collate = make_collate(name, tok, max_len, device)
    train_docs = load_docs(args.train)
    dev_docs = load_docs(args.dev)
    test_docs = load_docs(args.test)
    train_loader = DataLoader(DocDS(train_docs), batch_size=bs, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(DocDS(dev_docs), batch_size=bs, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(DocDS(test_docs), batch_size=bs, shuffle=False, collate_fn=collate)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.time()
    run_split(model, train_loader, True, device)
    run_split(model, dev_loader, False, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_sec = time.time() - t0
    peak = gpu_peak_mb()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    n_test = run_split(model, test_loader, False, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_sec = time.time() - t1
    train_sec = epoch_sec * args.epochs
    row = {
        "model": name,
        "total_params": total,
        "trainable_params": trainable,
        "total_params_m": round(total / 1e6, 2),
        "trainable_params_m": round(trainable / 1e6, 2),
        "epoch_wall_sec": round(epoch_sec, 1),
        "training_time_sec": round(train_sec, 1),
        "training_time_min": round(train_sec / 60.0, 1),
        "peak_gpu_mem_mb": None if peak is None else round(peak, 1),
        "test_n": n_test,
        "infer_test_sec": round(infer_sec, 2),
        "infer_ms_per_ex": round(1000.0 * infer_sec / max(1, n_test), 2),
        "device": str(device),
        "batch_size": bs,
        "max_length": max_len,
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    print(
        "TABLE | %s | Trainable: %.2fM | Time: %.1f min | Peak GPU: %s | Infer: %.2f ms/ex | bs=%d L=%d"
        % (
            name,
            row["trainable_params_m"],
            row["training_time_min"],
            ("%.1f MB" % peak) if peak is not None else "n/a",
            row["infer_ms_per_ex"],
            bs,
            max_len,
        ),
        flush=True,
    )
    del model, bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=["roberta", "branch_bert", "glan", "sitpcl", "all"])
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--dev", default=DEFAULT_DEV)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--bert-dir", default="")
    p.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE)
    p.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="0 = paper cap per model (512). Positive value overrides all models.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--cuda-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:%d" % args.cuda_index if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_index)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("need transformers: pip install transformers")
    except RuntimeError as e:
        raise SystemExit(
            "transformers is too new for this torch (%s): %s\n"
            "Do NOT upgrade torch. Use a venv that keeps GPU torch:\n"
            "  python3.9 -m venv --system-site-packages $HOME/venvs/baseline-eff\n"
            "  source $HOME/venvs/baseline-eff/bin/activate\n"
            "  python -m pip install 'transformers==4.46.3'\n"
            "  export HF_ENDPOINT=https://hf-mirror.com\n"
            "  python scripts/measure_baseline_efficiency.py --model all "
            "--bert-dir hfl/chinese-roberta-wwm-ext --cuda-index 0\n"
            % (torch.__version__, e)
        )

    bert_name = resolve_bert(args.bert_dir)
    print(
        "bert_dir=%s device=%s batch_size=%d max_length=%s"
        % (
            bert_name,
            device,
            args.batch_size,
            args.max_length if args.max_length > 0 else "paper/512",
        ),
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(bert_name)
    names = ["roberta", "branch_bert", "glan", "sitpcl"] if args.model == "all" else [args.model]
    rows = []
    for name in names:
        rows.append(measure_one(name, args, device, tok, bert_name))
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
