"""Qwen3 encoder that runs on PyTorch 1.13 (no torch.compiler / SDPA / DTensor).

Loads the same Qwen3-Embedding safetensors as HuggingFace Qwen3Model, then
does causal eager attention + last-token pooling. Tokenizer still comes from
transformers; this file only replaces the blocked modeling_qwen3 import.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv
        self.scale = self.head_dim ** -0.5
        bias = bool(cfg.attention_bias)
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_bias):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + attn_bias
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Layer(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_bias):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_bias)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3LegacyEncoder(nn.Module):
    """HF-compatible enough for last-token embedding: .config.hidden_size and last_hidden_state."""

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen3Layer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        inv = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def _rope(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.to(dtype=x.dtype).cos(), emb.to(dtype=x.dtype).sin()

    def _attn_bias(self, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t = attention_mask.shape
        pad = (1.0 - attention_mask[:, None, None, :].float()) * -1e4
        causal = torch.triu(torch.ones(t, t, device=attention_mask.device), diagonal=1) * -1e4
        return pad + causal

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = self.embed_tokens(input_ids)
        b, t, _ = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        cos, sin = self._rope(x, pos)
        bias = self._attn_bias(attention_mask).to(dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, bias)
        x = self.norm(x)
        return SimpleNamespace(last_hidden_state=x)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        dtype: torch.dtype,
        device: str,
    ) -> "Qwen3LegacyEncoder":
        path = Path(model_dir)
        raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
        head_dim = int(raw.get("head_dim") or int(raw["hidden_size"]) // int(raw["num_attention_heads"]))
        cfg = SimpleNamespace(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw.get("num_key_value_heads", raw["num_attention_heads"])),
            head_dim=head_dim,
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 1000000.0)),
            attention_bias=bool(raw.get("attention_bias", False)),
        )
        old_dtype = torch.get_default_dtype()
        if dtype in (torch.float16, torch.bfloat16):
            torch.set_default_dtype(dtype)
        try:
            model = cls(cfg)
        finally:
            torch.set_default_dtype(old_dtype)
        state = _load_weight_dict(path, dtype=dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        needed_missing = [k for k in missing if not k.startswith("inv_freq")]
        if needed_missing:
            raise RuntimeError("Qwen3 legacy load missing weights: %s" % needed_missing[:12])
        print(
            "legacy Qwen3 encoder: layers=%d hidden=%d unexpected=%d"
            % (cfg.num_hidden_layers, cfg.hidden_size, len(unexpected)),
            flush=True,
        )
        return model.to(device=device, dtype=dtype)


def _load_weight_dict(model_dir: Path, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    index = model_dir / "model.safetensors.index.json"
    shards = []
    if index.is_file():
        spec = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(spec["weight_map"].values()))
    elif (model_dir / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    state: Dict[str, torch.Tensor] = {}
    if shards:
        from safetensors.torch import load_file
        for name in shards:
            print("loading %s" % name, flush=True)
            chunk = load_file(str(model_dir / name), device="cpu")
            for key, value in chunk.items():
                state[_strip_key(key)] = value.to(dtype=dtype)
            del chunk
    else:
        bins = sorted(model_dir.glob("pytorch_model*.bin"))
        if not bins:
            raise FileNotFoundError("no safetensors or pytorch_model*.bin in %s" % model_dir)
        for bin_path in bins:
            print("loading %s" % bin_path.name, flush=True)
            chunk = torch.load(str(bin_path), map_location="cpu")
            for key, value in chunk.items():
                state[_strip_key(key)] = value.to(dtype=dtype)
            del chunk
    state.pop("lm_head.weight", None)
    state.pop("lm_head.bias", None)
    return state


def _strip_key(key: str) -> str:
    if key.startswith("lm_head."):
        return key
    if key.startswith("model."):
        return key[len("model.") :]
    return key
