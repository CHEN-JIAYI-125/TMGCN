"""Frozen Qwen node/target features with unchanged global IDs and graph topology.

Never encodes labels, fits on test data, or broadcasts a conversation vector.
Run from the repository root. Requires transformers>=4.51, torch, numpy, PyYAML.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sentence_rows(split_records):
    """Validate the existing global-id mapping before loading expensive weights."""
    rows = {}
    for records in split_records:
        for rec in records:
            sentences, ids = rec['sentences'], rec['global_ids']
            if len(sentences) != len(ids):
                raise ValueError('sentences/global_ids length mismatch')
            for gid, text in zip(ids, sentences):
                if not isinstance(gid, int) or isinstance(gid, bool) or gid < 0:
                    raise ValueError('global_ids must be nonnegative integers')
                if gid in rows and rows[gid] != text:
                    raise ValueError(f'Conflicting sentence for global_id={gid}')
                rows[gid] = text
    if not rows:
        raise ValueError('No utterances found')
    return rows


def target_rows(template):
    targets, texts = [], []
    for target, info in template.items():
        views = [target] + [info.get(k, '').strip() for k in
                             ('description', 'favor_reason', 'against_reason')]
        if not all(views):
            raise ValueError(f'Incomplete target views: {target}')
        targets.append(target)
        texts.extend(views)
    return targets, texts


def _patch_torch_distributed_tensor():
    """Qwen3 transformers imports torch.distributed.tensor (PyTorch >= 2.5)."""
    import sys
    import types

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for key, val in attrs.items():
            setattr(m, key, val)
        sys.modules[name] = m
        return m

    dummy_cls = type('DTensorDummy', (), {'__init__': lambda *a, **k: None})
    try:
        import torch.distributed.tensor  # noqa: F401
        return
    except Exception:
        pass
    try:
        import torch.distributed._tensor as _dt  # type: ignore
        sys.modules['torch.distributed.tensor'] = _dt
        import torch.distributed as dist
        dist.tensor = _dt
        return
    except Exception:
        pass
    tensor = _mod(
        'torch.distributed.tensor',
        DTensor=dummy_cls,
        Shard=dummy_cls,
        Replicate=dummy_cls,
        Partial=dummy_cls,
        Placement=dummy_cls,
        DeviceMesh=dummy_cls,
        distribute_tensor=lambda *a, **k: None,
        distribute_module=lambda *a, **k: None,
    )
    _mod(
        'torch.distributed.tensor.placement_types',
        Placement=dummy_cls, Shard=dummy_cls, Replicate=dummy_cls, Partial=dummy_cls,
    )
    _mod('torch.distributed.tensor.device_mesh', DeviceMesh=dummy_cls)
    try:
        import torch.distributed as dist
        dist.tensor = tensor
    except Exception:
        pass


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml

    _patch_torch_distributed_tensor()
    from transformers import AutoModel, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/config_model5.yaml')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--target-template', default='datasets/target_knowledge_template.json')
    p.add_argument('--allow-download', action='store_true',
                   help='允许从 HuggingFace 拉权重；默认只读本地目录')
    a = p.parse_args()
    if a.batch_size < 1 or a.max_length < 8:
        p.error('batch-size must be positive and max-length >= 8')
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing cache: {out}')
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    split_paths = [Path(cfg[f'{s}_path']).resolve() for s in ('train', 'dev', 'test')]
    records = [json.loads(path.read_text(encoding='utf-8')) for path in split_paths]
    rows = sentence_rows(records)
    targets, views = target_rows(json.loads(Path(a.target_template).read_text(encoding='utf-8')))
    missing = {rec['target'] for split in records for rec in split} - set(targets)
    if missing and cfg.get('use_grounding', False):
        raise ValueError(f'Target template missing {len(missing)} data targets')
    out.mkdir(parents=True, exist_ok=True)
    print(f'Unique nodes={len(rows)}; target views={len(views)}', flush=True)
    local_only = not a.allow_download
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=local_only, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[a.dtype]
    model = AutoModel.from_pretrained(
        a.model, torch_dtype=dtype, local_files_only=local_only, trust_remote_code=True,
    ).to(a.device).eval()
    model.requires_grad_(False)
    dim = model.config.hidden_size
    instruction = 'Represent the meaning and stance of this text for stance classification.'
    truncated = 0

    def encode(texts):
        nonlocal truncated
        prompts = [f'Instruct: {instruction}\nQuery: {text}' for text in texts]
        lengths = tokenizer(prompts, truncation=False)['input_ids']
        truncated += sum(len(x) > a.max_length for x in lengths)
        batch = tokenizer(prompts, padding=True, truncation=True,
                          max_length=a.max_length, return_tensors='pt').to(a.device)
        with torch.inference_mode():
            hidden = model(**batch, use_cache=False).last_hidden_state[:, -1]
            result = F.normalize(hidden.float(), p=2, dim=-1).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Non-finite Qwen embeddings')
        return result

    matrix = np.lib.format.open_memmap(out / 'embeddings.npy', mode='w+', dtype='float32',
                                       shape=(max(rows) + 1, dim))
    matrix[:] = 0  # holes in global IDs are unused; never remap IDs
    ids = sorted(rows)
    for start in range(0, len(ids), a.batch_size):
        chunk = ids[start:start + a.batch_size]
        matrix[chunk] = encode([rows[i] for i in chunk])
        if start % (a.batch_size * 50) == 0:
            matrix.flush()
            print(f'Nodes {start + len(chunk)}/{len(ids)}', flush=True)
    matrix.flush()
    prior = np.empty((len(views), dim), dtype=np.float32)
    for start in range(0, len(views), a.batch_size):
        prior[start:start + a.batch_size] = encode(views[start:start + a.batch_size])
    np.save(out / 'target_knowledge.npy', prior.reshape(len(targets), 4, dim))
    (out / 'target_knowledge_index.json').write_text(
        json.dumps(dict(zip(targets, range(len(targets)))), ensure_ascii=False), encoding='utf-8')
    for split, path in zip(('train', 'dev', 'test'), split_paths):
        cfg[f'{split}_path'] = str(path)
    cfg.update(embedding_npy=str(out / 'embeddings.npy'),
               target_knowledge_npy=str(out / 'target_knowledge.npy'), input_size=dim,
               result_root=str(out / 'results'), time='qwen_nodes')
    (out / 'train.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    manifest = dict(model=str(Path(a.model).resolve()), dimension=dim, nodes=len(rows),
                    pooling='last_token', normalized=True, truncated=truncated, args=vars(a),
                    data_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in split_paths})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'COMPLETE: {out}/train.yaml; truncated texts={truncated}', flush=True)


if __name__ == '__main__':
    main()
