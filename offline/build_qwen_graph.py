"""Frozen Qwen node/target features, stance-conditioned by conversation + target.

Same sentence text (same global_id) can take opposite stances under different
targets, e.g. 「日本动漫文化」 vs 「反对日本动漫文化」. RoBERTa rows keyed only
by global_id collapse those cases. This encoder keeps graph IDs/topology, but
looks up features by (target, global_id) and puts the current conversation,
the current utterance, and the target into the Qwen prompt.

Never encodes gold Favor/Against/Neutral labels. Qwen stays frozen.

Run from the repository root. Requires transformers>=4.51, torch, numpy, PyYAML.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STANCE_INSTRUCTION = (
    "当前会话中，当前话语对目标的立场是什么？"
    "编码该话语在这段会话里的立场含义。"
    "同一段会话面对不同目标时，立场可能相反。"
)

TARGET_INSTRUCTION = "编码该立场对象及其支持或反对观点，用于会话立场分类。"


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


def _speaker_tag(speakers, turn):
    if turn >= len(speakers):
        return 'speaker_UNK'
    raw = speakers[turn]
    if raw is None or raw == '':
        return 'speaker_UNK'
    return f'speaker_{raw}'


def build_utterance_prompt(target, sentences, speakers, turn):
    """Qwen3-Embedding Instruct/Query; conversation + current utterance, no gold label."""
    lines = []
    for i, sent in enumerate(sentences):
        mark = ' [当前话语]' if i == turn else ''
        lines.append(f'[{i + 1}] {_speaker_tag(speakers, i)}:{mark} {sent}')
    query = (
        f'目标：{target}\n'
        f'\n'
        f'当前会话：\n'
        f'{chr(10).join(lines)}\n'
        f'\n'
        f'当前话语：\n'
        f'{sentences[turn]}'
    )
    return f'Instruct: {STANCE_INSTRUCTION}\nQuery: {query}'


def build_target_prompt(text):
    return f'Instruct: {TARGET_INSTRUCTION}\nQuery: {text}'


def collect_utterance_units(split_records):
    """One unit per (target, global_id); first occurrence wins (train→dev→test, doc id)."""
    sentence_rows(split_records)
    seen = {}
    units = []
    for records in split_records:
        recs = sorted(
            records,
            key=lambda r: int(r.get('id', r.get('doc_id', -1))),
        )
        for rec in recs:
            target = str(rec.get('target', '') or '')
            sentences = [str(s) for s in rec['sentences']]
            ids = rec['global_ids']
            speakers = rec.get('speakers') or []
            for turn, gid in enumerate(ids):
                key = (target, int(gid))
                if key in seen:
                    continue
                seen[key] = len(units)
                units.append({
                    'target': target,
                    'gid': int(gid),
                    'prompt': build_utterance_prompt(target, sentences, speakers, turn),
                })
    if not units:
        raise ValueError('No utterances found')
    return units


def units_to_index(units):
    index = {}
    for row, unit in enumerate(units):
        index.setdefault(unit['target'], {})[str(unit['gid'])] = row
    return index


def _install_qwen_torch_shims():
    """Fill PyTorch 2.5-only modules so Qwen3 transformers can import on 2.1–2.4.

    Do not return early if torch.distributed._tensor exists: that module is not
    a package and lacks placement_types / parallel, which still crashes import.
    """
    import sys
    import types

    dummy_cls = type('DTensorDummy', (), {'__init__': lambda *a, **k: None})

    def ensure(name, is_package=False, **attrs):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        if is_package and not hasattr(mod, '__path__'):
            mod.__path__ = []
        parent, _, child = name.rpartition('.')
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, mod)
        for key, val in attrs.items():
            if not hasattr(mod, key):
                setattr(mod, key, val)
        return mod

    ensure('torch.distributed', is_package=True)
    tensor = ensure(
        'torch.distributed.tensor',
        is_package=True,
        DTensor=dummy_cls,
        Shard=dummy_cls,
        Replicate=dummy_cls,
        Partial=dummy_cls,
        Placement=dummy_cls,
        DeviceMesh=dummy_cls,
        TensorMeta=dummy_cls,
        DTensorSpec=dummy_cls,
        distribute_tensor=lambda *a, **k: None,
        distribute_module=lambda *a, **k: None,
        ones=lambda *a, **k: None,
        zeros=lambda *a, **k: None,
        empty=lambda *a, **k: None,
    )
    ensure(
        'torch.distributed.tensor.placement_types',
        Placement=dummy_cls, Shard=dummy_cls, Replicate=dummy_cls, Partial=dummy_cls,
    )
    ensure('torch.distributed.tensor.device_mesh', DeviceMesh=dummy_cls)
    ensure(
        'torch.distributed.tensor.parallel',
        ColwiseParallel=dummy_cls,
        RowwiseParallel=dummy_cls,
        SequenceParallel=dummy_cls,
        PrepareModuleInput=dummy_cls,
        PrepareModuleOutput=dummy_cls,
        parallelize_module=lambda *a, **k: None,
    )
    ensure('torch.distributed.tensor._dtensor')
    ensure('torch.distributed.tensor._utils')
    try:
        import torch.distributed as dist
        dist.tensor = tensor
    except Exception:
        pass

    try:
        import torch.nn.attention.flex_attention  # noqa: F401
    except Exception:
        ensure('torch.nn.attention', is_package=True)
        ensure(
            'torch.nn.attention.flex_attention',
            BlockMask=dummy_cls,
            create_block_mask=lambda *a, **k: None,
            flex_attention=lambda *a, **k: None,
            _DEFAULT_SPARSE_BLOCK_SIZE=128,
        )


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml

    _install_qwen_torch_shims()
    import transformers
    from transformers import AutoModel, AutoTokenizer
    print(f'torch={torch.__version__} transformers={transformers.__version__}', flush=True)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/config_model5.yaml')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--target-template', default='datasets/target_knowledge_template.json')
    p.add_argument('--allow-download', action='store_true',
                   help='允许从 HuggingFace 拉权重；默认只读本地目录')
    p.add_argument('--overwrite', action='store_true',
                   help='允许覆盖已有 cache 目录中的 npy/json')
    a = p.parse_args()
    if a.batch_size < 1 or a.max_length < 8:
        p.error('batch-size must be positive and max-length >= 8')
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f'Refusing to overwrite existing cache: {out} (pass --overwrite)')
    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    split_paths = [Path(cfg[f'{s}_path']).resolve() for s in ('train', 'dev', 'test')]
    records = [json.loads(path.read_text(encoding='utf-8')) for path in split_paths]
    units = collect_utterance_units(records)
    targets, views = target_rows(json.loads(Path(a.target_template).read_text(encoding='utf-8')))
    missing = {rec['target'] for split in records for rec in split} - set(targets)
    if missing and cfg.get('use_grounding', False):
        raise ValueError(f'Target template missing {len(missing)} data targets')
    out.mkdir(parents=True, exist_ok=True)
    n_targets = len({u['target'] for u in units})
    print(f'Unique (target,gid)={len(units)}; targets={n_targets}; target views={len(views)}', flush=True)
    print(f'Prompt preview:\n{units[0]["prompt"]}', flush=True)
    local_only = not a.allow_download
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=local_only, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[a.dtype]
    torch_ver = torch.__version__.split('+')[0].split('.')
    torch_tuple = (int(torch_ver[0]), int(torch_ver[1]))
    if dtype is torch.bfloat16 and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 8:
            print('GPU compute capability %s does not support bf16; using fp16' % (cap,), flush=True)
            dtype = torch.float16
    if torch_tuple < (2, 0):
        print(
            'torch %s cannot import modeling_qwen3; using legacy eager encoder. '
            'Prefer --batch-size 1 --dtype fp16' % torch.__version__,
            flush=True,
        )
        try:
            from qwen3_legacy import Qwen3LegacyEncoder
        except ImportError:
            from offline.qwen3_legacy import Qwen3LegacyEncoder
        model = Qwen3LegacyEncoder.from_pretrained(a.model, dtype=dtype, device=a.device)
    else:
        try:
            model = AutoModel.from_pretrained(
                a.model, torch_dtype=dtype, local_files_only=local_only, trust_remote_code=True,
                attn_implementation='eager',
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                a.model, torch_dtype=dtype, local_files_only=local_only, trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                '加载 Qwen3-Embedding 失败。当前 torch=%s transformers=%s。原始错误: %r'
                % (torch.__version__, transformers.__version__, exc)
            ) from exc
        model = model.to(a.device)
    model = model.eval()
    model.requires_grad_(False)
    dim = model.config.hidden_size
    truncated = 0

    def encode(texts):
        nonlocal truncated
        lengths = tokenizer(texts, truncation=False)['input_ids']
        truncated += sum(len(x) > a.max_length for x in lengths)
        batch = tokenizer(texts, padding=True, truncation=True,
                          max_length=a.max_length, return_tensors='pt').to(a.device)
        with torch.inference_mode():
            hidden = model(**batch, use_cache=False).last_hidden_state[:, -1]
            result = F.normalize(hidden.float(), p=2, dim=-1).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Non-finite Qwen embeddings')
        return result

    matrix = np.lib.format.open_memmap(
        out / 'embeddings.npy', mode='w+', dtype='float32', shape=(len(units), dim),
    )
    for start in range(0, len(units), a.batch_size):
        chunk = units[start:start + a.batch_size]
        matrix[start:start + len(chunk)] = encode([u['prompt'] for u in chunk])
        if start % (a.batch_size * 50) == 0:
            matrix.flush()
            print(f'Nodes {start + len(chunk)}/{len(units)}', flush=True)
    matrix.flush()
    (out / 'embeddings_index.json').write_text(
        json.dumps(units_to_index(units), ensure_ascii=False), encoding='utf-8',
    )
    prior = np.empty((len(views), dim), dtype=np.float32)
    target_prompts = [build_target_prompt(text) for text in views]
    for start in range(0, len(target_prompts), a.batch_size):
        prior[start:start + a.batch_size] = encode(target_prompts[start:start + a.batch_size])
    np.save(out / 'target_knowledge.npy', prior.reshape(len(targets), 4, dim))
    (out / 'target_knowledge_index.json').write_text(
        json.dumps(dict(zip(targets, range(len(targets)))), ensure_ascii=False), encoding='utf-8')
    for split, path in zip(('train', 'dev', 'test'), split_paths):
        cfg[f'{split}_path'] = str(path)
    cfg.update(
        embedding_npy=str(out / 'embeddings.npy'),
        target_knowledge_npy=str(out / 'target_knowledge.npy'),
        input_size=dim,
        result_root=str(out / 'results'),
        time='qwen_stance',
    )
    (out / 'train.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    manifest = dict(
        model=str(Path(a.model).resolve()),
        dimension=dim,
        nodes=len(units),
        keyed_by='target,global_id',
        instruction=STANCE_INSTRUCTION,
        pooling='last_token',
        normalized=True,
        truncated=truncated,
        gold_labels_in_prompt=False,
        args=vars(a),
        data_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in split_paths},
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'COMPLETE: {out}/train.yaml; truncated texts={truncated}', flush=True)


if __name__ == '__main__':
    main()
