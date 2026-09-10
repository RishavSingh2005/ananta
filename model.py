"""
Ananta-Tiny: a small decoder-only Transformer, implemented from scratch on top
of PyTorch (PyTorch is used the same way NumPy is -- as a math/autodiff
library, not as a source of pretrained weights). Every parameter in this file
is randomly initialized; nothing is loaded from a pretrained checkpoint.

Design choices and why:
  - RMSNorm instead of LayerNorm: one fewer reduction (no mean-subtraction),
    same stabilizing effect in practice. Used by Llama/Mistral/Gemma.
  - Rotary position embeddings (RoPE) instead of learned absolute positions:
    encodes relative position directly in the attention dot product, and
    generalizes better to sequence lengths seen rarely during training.
  - Grouped-query attention (GQA)-ready: n_kv_heads can be set below n_heads
    to cut KV memory/compute. At this tiny scale we default n_kv_heads =
    n_heads (plain multi-head attention); the mechanism is here for when you
    scale up.
  - SwiGLU feed-forward instead of a plain GELU MLP: a gated activation that
    consistently outperforms plain GELU-MLP at matched parameter count
    (used in Llama/PaLM/Mistral).
  - F.scaled_dot_product_attention: lets PyTorch pick a fused/flash attention
    kernel when the platform supports one, instead of a naive O(T^2) python
    implementation.
  - Weight tying: the input embedding and output (unembedding) matrix share
    weights, a standard param-saving trick with no accuracy cost.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnantaConfig:
    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8       # set < n_heads for grouped-query attention
    max_seq_len: int = 512
    ffn_mult: float = 8 / 3   # SwiGLU hidden-size multiplier (matches ~4x GELU-MLP param count)
    dropout: float = 0.1
    rope_theta: float = 10000.0
    tie_weights: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, n_heads, T, head_dim) ; cos/sin: (max_seq_len, head_dim/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[2], :].to(x.dtype)
    sin = sin[None, None, : x.shape[2], :].to(x.dtype)
    out_even = x1 * cos - x2 * sin
    out_odd = x1 * sin + x2 * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AnantaConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: AnantaConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        hidden = ((hidden + 63) // 64) * 64  # round up to a multiple of 64 (kernel-friendly)
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: AnantaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AnantaTransformer(nn.Module):
    def __init__(self, cfg: AnantaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = build_rope_cache(cfg.max_seq_len, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-writing projections, so the
        # residual stream's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                mask = (cum_probs - probs) > top_p
                sorted_logits[mask] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx
