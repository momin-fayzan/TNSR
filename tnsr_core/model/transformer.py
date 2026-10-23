"""Decoder-only Transformer (GPT-style) implemented from scratch on PyTorch tensors.

Architecture: token embedding (+ learned positional embedding or rotary
position encoding) -> N pre-norm blocks [norm -> causal multi-head
self-attention -> residual, norm -> MLP -> residual] -> final norm -> vocab
projection (weight-tied to the token embedding by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tnsr_core.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)


def make_norm(cfg: ModelConfig) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.n_embd)
    return nn.LayerNorm(cfg.n_embd, bias=cfg.bias)


def rope_tables(
    head_dim: int, max_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels. x: (B, H, T, hd); cos/sin: (T, hd/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


@dataclass
class LayerCache:
    """Key/value cache for one attention layer during incremental decoding."""

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.size(2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.attn_impl = cfg.attn_impl
        self.causal = cfg.causal
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: LayerCache | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if rope is not None:
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)

        past = 0
        if cache is not None:
            past = cache.length
            if cache.k is not None and cache.v is not None:
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            cache.k, cache.v = k, v

        S = k.size(2)  # total key length (past + current)
        dropout_p = self.dropout if self.training else 0.0
        if not self.causal:
            if self.attn_impl == "sdpa":
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                att = self.attn_drop(F.softmax(att.float(), dim=-1).to(q.dtype))
                y = att @ v
        elif self.attn_impl == "sdpa":
            if past == 0:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            elif T == 1:
                # A single new query may attend to every cached position.
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                mask = causal_mask(T, S, x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~causal_mask(T, S, x.device), float("-inf"))
            att = self.attn_drop(F.softmax(att.float(), dim=-1).to(q.dtype))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


def causal_mask(T: int, S: int, device: torch.device) -> torch.Tensor:
    """Boolean (T, S) mask; query i (at absolute position S-T+i) sees keys <= its position."""
    q_pos = torch.arange(S - T, S, device=device).unsqueeze(1)
    k_pos = torch.arange(S, device=device).unsqueeze(0)
    return k_pos <= q_pos


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None, cache=None):
        x = x + self.attn(self.norm1(x), rope=rope, cache=cache)
        x = x + self.mlp(self.norm2(x))
        return x


class TNSRModel(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = (
            nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_encoding == "learned" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg)
        if cfg.num_classes:
            # Classifier head on the last position (e.g. win-probability bins for a chess move).
            self.lm_head = nn.Linear(cfg.n_embd, cfg.num_classes, bias=True)
        else:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.tok_emb.weight
        if cfg.pos_encoding == "rope":
            cos, sin = rope_tables(cfg.head_dim, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 style) keeps early activations stable.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def new_cache(self) -> list[LayerCache]:
        return [LayerCache() for _ in self.blocks]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: list[LayerCache] | None = None,
        last_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits, loss). ``idx``: (B, T) token ids.

        With ``cache``, ``idx`` holds only the new tokens and positions continue
        from the cached length. ``last_only`` computes logits for the final
        position only (cheaper during generation).
        """
        B, T = idx.shape
        if cache is not None and not self.config.causal:
            raise ValueError("KV caching requires a causal model")
        start = cache[0].length if cache else 0
        if start + T > self.config.block_size:
            raise ValueError(
                f"Sequence length {start + T} exceeds block_size {self.config.block_size}"
            )
        x = self.tok_emb(idx)
        rope = None
        if self.pos_emb is not None:
            pos = torch.arange(start, start + T, device=idx.device)
            x = x + self.pos_emb(pos)
        else:
            rope = (self.rope_cos[start : start + T], self.rope_sin[start : start + T])
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, rope=rope, cache=cache[i] if cache else None)
        x = self.norm_f(x)
        if self.config.num_classes:
            logits = self.lm_head(x[:, -1, :])  # (B, num_classes)
            loss = F.cross_entropy(logits.float(), targets) if targets is not None else None
            return logits, loss
        if last_only and targets is None:
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def optimizer_groups(self, weight_decay: float) -> list[dict]:
        """Apply weight decay to matrices only (not norms, biases or embeddings' 1-D params)."""
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        seen: set[int] = set()
        for p in self.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def estimate_flops_per_token(self) -> int:
        """Approximate training FLOPs per token (6N + attention), for throughput reporting."""
        cfg = self.config
        n = self.num_parameters(non_embedding=True)
        attn = 12 * cfg.n_layer * cfg.n_embd * cfg.block_size
        return 6 * n + attn
