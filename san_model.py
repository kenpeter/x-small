"""Simple Attention Network (SAN) — PyTorch port of cactus-compute/needle architecture.

Faithful port of needle/model/architecture.py (arXiv:2607.18363) from JAX/Flax
to PyTorch, sized to the needle2 scale (~45M params). Components, 1:1 with the
JAX source:
  - ZCRMSNorm         (zero-centered RMSNorm)
  - HadamardMLP       (fixed Walsh-Hadamard transform MLP — no dense FFN weights)
  - MultiHeadAttention(GQA + q/k RMSNorm + RoPE + learned gate)
  - Engram            (hashed n-gram KV memory tables, conv taps, injected at sites)
  - Multi-Lane Hyper-Connections (MHC) with Sinkhorn routing across 4 residual lanes
  - MTP               (multi-token-prediction head)
  - Tied embeddings + causal/packing mask
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193
_STD = 0.02
_LANE_OFF = 8.0
_POST_OFF = -4.0


@dataclass
class SANConfig:
    vocab_size: int = 49152
    d_model: int = 384
    attn_dim: int = 0
    num_heads: int = 8
    num_kv_heads: int = 4
    num_layers: int = 27
    max_seq_len: int = 2048
    pad_token_id: int = 0
    rope_theta: float = 100000.0
    dtype: str = "bfloat16"
    engram_orders: tuple = (2, 3)
    engram_heads: int = 0
    engram_slots: int = 8192
    engram_layers: tuple = (2, 15)
    mhc_lanes: int = 4

    def __post_init__(self):
        self.attn_dim = self.attn_dim or self.d_model
        self.engram_layers = tuple(self.engram_layers)


def _walsh_matrix(n):
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def engram_geometry(cfg):
    orders = tuple(cfg.engram_orders)
    heads = cfg.engram_heads or max(1, cfg.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = cfg.d_model // (len(orders) * heads)
    return orders, heads, sub_dim


def _shift_right(x, offset):
    """Pad the time axis (dim 1) on the left by `offset`, then truncate back to
    original length — matches JAX `jnp.pad(x, pad)[:, : x.shape[1]]`."""
    if offset == 0:
        return x
    if x.ndim == 2:
        return F.pad(x, (offset, 0))[:, : x.shape[1]]
    T = x.shape[1]
    return F.pad(x, (0, 0, offset, 0))[:, :T, :]  # pad dim1 by offset, truncate to T


def _mask_diag(mask, offset):
    """mask: (B,1,T,T) -> (B,T) diagonal at -offset (matches JAX jnp.diagonal)."""
    m = mask[:, 0]  # (B,T,T)
    T = m.shape[-1]
    if offset >= T:
        return torch.zeros(m.shape[:-2] + (T,), dtype=m.dtype, device=m.device)
    d = torch.diagonal(m, offset=-offset, dim1=-2, dim2=-1)  # (B,T-offset)
    if offset == 0:
        return d
    return F.pad(d, (offset, 0))  # (B,T)


def engram_indices(tokens, orders, heads, slots):
    u = tokens.to(torch.int64)
    B, T = u.shape
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = torch.full((B, T), seed, dtype=torch.int64, device=tokens.device)
            for j in range(order):
                acc = (acc ^ _shift_right(u, j)) * _ENGRAM_PRIME
            acc = (acc & 0xFFFFFFFF) ^ ((acc & 0xFFFFFFFF) >> 15)
            idx.append((acc % slots).to(torch.int64))
    return torch.stack(idx, dim=-1)


def make_causal_mask(seq_len, device=None):
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))[None, None]


def make_causal_packing_mask(seg_ids, prefix=None, window=0):
    T = seg_ids.shape[1]
    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=seg_ids.device))
    seg = seg_ids[:, :, None] == seg_ids[:, None, :]
    block = seg & (seg_ids[:, :, None] > 0)
    mask = block & causal[None, :, :]
    if window:
        pos = torch.arange(T, device=seg_ids.device)
        recent = (pos[:, None] - pos[None, :]) < window
        sink = torch.zeros_like(seg_ids, dtype=torch.bool) if prefix is None else (prefix > 0)
        mask = mask & (recent[None, :, :] | sink[:, None, :])
    return mask[:, None, :, :]


class ZCRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1 + self.scale) * x / rms).to(self.dtype)


def _sinkhorn(logits, iters=20):
    log_K = logits.float()
    for _ in range(iters):
        log_K = log_K - torch.logsumexp(log_K, dim=-1, keepdim=True)
        log_K = log_K - torch.logsumexp(log_K, dim=-2, keepdim=True)
    return torch.exp(log_K)


class HadamardMLP(nn.Module):
    def __init__(self, d_model, dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        n = 1 << (d_model - 1).bit_length()
        self.n = n
        self.register_buffer("H", _walsh_matrix(n).to(dtype))
        self.d1 = nn.Parameter(torch.ones(n).to(dtype))
        self.d2 = nn.Parameter(torch.ones(n).to(dtype))
        self.d3 = nn.Parameter(torch.full((n,), 0.02).to(dtype))

    def forward(self, x):
        n, d = self.n, self.d_model
        z = F.pad(x, (0, n - d)) if n > d else x
        z = self.d1 * z.to(self.dtype)
        z = z @ self.H
        z = F.silu(self.d2 * z) @ self.H
        return (self.d3 * z)[..., :d]


class Engram(nn.Module):
    def __init__(self, d_model, num_tables, slots, sub_dim, num_layers, conv_dilation,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_tables = num_tables
        self.slots = slots
        self.sub_dim = sub_dim
        self.conv_dilation = conv_dilation
        self.dtype = dtype
        in_dim = num_tables * sub_dim
        self.embedding = nn.Parameter(torch.empty(num_tables, slots, sub_dim).normal_(0, _STD)
                                      .to(dtype))
        self.key_proj = nn.Linear(in_dim, d_model, bias=False, dtype=dtype)
        self.value_proj = nn.Linear(in_dim, d_model, bias=False, dtype=dtype)
        taps = torch.zeros(ENGRAM_CONV_TAPS, d_model)
        taps[0] = 1.0
        self.taps = nn.Parameter(taps.to(dtype))
        nn.init.normal_(self.key_proj.weight, 0, _STD)
        nn.init.normal_(self.value_proj.weight, 0, _STD / math.sqrt(2 * num_layers))

    def forward(self, indices, ngram_ok, tap_ok):
        B, T, _ = indices.shape
        tables = self.embedding
        fetched = tables[torch.arange(self.num_tables, device=indices.device), indices]
        if ngram_ok is not None:
            fetched = fetched * ngram_ok.unsqueeze(-1)
        e = fetched.reshape(B, T, self.num_tables * self.sub_dim).to(self.dtype)
        k = self.key_proj(e)
        v = self.value_proj(e)
        if tap_ok is not None:
            v = sum(self.taps[j] * _shift_right(v, j * self.conv_dilation) * tap_ok[j].unsqueeze(-1)
                    for j in range(ENGRAM_CONV_TAPS))
        else:
            v = sum(self.taps[j] * _shift_right(v, j * self.conv_dilation)
                    for j in range(ENGRAM_CONV_TAPS))
        return k, v


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def precompute_rope_freqs(head_dim, seq_len, theta=100000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)
    return torch.cos(angles), torch.sin(angles)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, d_model, num_layers, dtype=torch.bfloat16,
                 attn_dim=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        attn_dim = attn_dim or d_model
        self.head_dim = attn_dim // num_heads
        res_std = _STD / math.sqrt(2 * num_layers)
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.q_norm = ZCRMSNorm(self.head_dim, dtype=dtype)
        self.k_norm = ZCRMSNorm(self.head_dim, dtype=dtype)
        self.gate_proj = nn.Linear(attn_dim, attn_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(attn_dim, d_model, bias=False, dtype=dtype)
        for w in (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.gate_proj.weight):
            nn.init.normal_(w, 0, _STD)
        nn.init.normal_(self.out_proj.weight, 0, res_std)

    def forward(self, x, mask=None, rope=None):
        B, T, _ = x.shape
        hd, nk = self.head_dim, self.num_kv_heads
        q = self.q_proj(x).view(B, T, self.num_heads, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, nk, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nk, hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        n_rep = self.num_heads // nk
        if n_rep > 1:
            k, v = k.repeat_interleave(n_rep, 1), v.repeat_interleave(n_rep, 1)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn.float(), dim=-1).to(self.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.num_heads * hd)
        out = out * torch.sigmoid(self.gate_proj(x))
        return self.out_proj(out.to(self.dtype))


def _rms_unit(x, eps=1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def probe_pool(cells, probes, keep=None, dtype=torch.bfloat16):
    """Pool hidden cells (B,T,L,d) with learned probes -> (B,K*d_out). Faithful port."""
    b, t, l, d = cells.shape
    cells = cells.reshape(b, t * l, d)
    if keep is not None:
        keep = keep.repeat(1, l)  # (B, T*L)
    scores = torch.einsum("bcd,kd->bkc", cells.float(), probes.float()) / math.sqrt(d)
    if keep is not None:
        scores = torch.where(keep[:, None, :] > 0, scores, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("bkc,bcd->bkd", w, cells.float()).reshape(b, -1).to(dtype)


class ContrastiveHead(nn.Module):
    """Learned-probe pooling + projection + temperature (tool retrieval)."""
    def __init__(self, d_model, out_dim, dtype=torch.bfloat16, probes=4, temp_init=0.07):
        super().__init__()
        self.dtype = dtype
        self.probes = nn.Parameter(torch.empty(probes, d_model).normal_(0, _STD))
        self.proj = nn.Linear(probes * d_model, out_dim, bias=False, dtype=dtype)
        nn.init.normal_(self.proj.weight, 0, _STD)
        self.log_temp = nn.Parameter(torch.tensor(math.log(temp_init)))

    def forward(self, cells, keep=None):
        pooled = probe_pool(cells, self.probes, keep, self.dtype)
        p = self.proj(pooled)
        denom = torch.sqrt(p.float().pow(2).sum(-1, keepdim=True) + 1e-12)
        return (p / denom.to(self.dtype)), self.log_temp


class ConfidenceHead(nn.Module):
    """Probe-pooled scalar logit (confidence gating)."""
    def __init__(self, d_model, dtype=torch.bfloat16, probes=8):
        super().__init__()
        self.dtype = dtype
        self.probes = nn.Parameter(torch.empty(probes, d_model).normal_(0, _STD))
        self.proj = nn.Linear(probes * d_model, 1, bias=True, dtype=dtype)
        nn.init.normal_(self.proj.weight, 0, _STD)

    def forward(self, cells, keep=None):
        pooled = probe_pool(cells, self.probes, keep, self.dtype)
        return self.proj(pooled)[..., 0].float()


class Block(nn.Module):
    def __init__(self, num_heads, num_kv_heads, d_model, num_layers, dtype=torch.bfloat16,
                 attn_dim=0):
        super().__init__()
        self.dtype = dtype
        self.norm1 = ZCRMSNorm(d_model, dtype=dtype)
        self.attn = MultiHeadAttention(num_heads, num_kv_heads, d_model, num_layers,
                                       dtype, attn_dim=attn_dim)
        self.post_attn_norm = ZCRMSNorm(d_model, dtype=dtype)
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.pre_hada_norm = ZCRMSNorm(d_model, dtype=dtype)
        self.hadamard_mlp = HadamardMLP(d_model, dtype=dtype)

    def forward(self, x, mask=None, rope=None, engram_kv=None, site_flags=None):
        if engram_kv is not None:
            ek, ev = engram_kv  # each (S, B, T, d)
            # site_flags: (S,) per-layer row — which sites this layer fires
            alpha = torch.sigmoid(
                torch.einsum("btd,sbtd->sbt", _rms_unit(x), _rms_unit(ek))
                / math.sqrt(x.shape[-1]))  # (S,B,T)
            x = x + torch.einsum("s,sbt,sbtd->btd",
                                 site_flags.float(), alpha.float(), ev.float()).to(x.dtype)
        skip = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask, rope=rope)
        x = self.post_attn_norm(x)
        x = skip + torch.sigmoid(self.attn_gate) * x
        skip = x
        x = self.pre_hada_norm(x)
        x = self.hadamard_mlp(x)
        return skip + x


class MHC(nn.Module):
    """Multi-Lane Hyper-Connections (JAX _ScanBody/Stack). Learned per-layer routing
    across `lanes` residual streams with Sinkhorn-normalized mixing."""
    def __init__(self, d_model, num_layers, lanes, dtype=torch.bfloat16):
        super().__init__()
        self.lanes = lanes
        nC = lanes * d_model
        phi = lambda: nn.Parameter(torch.empty(num_layers, nC, lanes).normal_(0, _STD))
        self.phi_pre = phi()
        self.phi_post = phi()
        self.phi_res = nn.Parameter(torch.empty(num_layers, nC, lanes * lanes).normal_(0, _STD))
        self.b_pre = nn.Parameter(torch.zeros(num_layers, lanes))
        self.b_post = nn.Parameter(torch.zeros(num_layers, lanes))
        self.b_res = nn.Parameter(torch.zeros(num_layers, lanes, lanes))
        # b_res identity init: 4 * eye(lanes) per layer
        with torch.no_grad():
            for L in range(num_layers):
                self.b_res[L] = 4.0 * torch.eye(lanes)
        self.a_pre = nn.Parameter(torch.full((num_layers,), 0.01))
        self.a_post = nn.Parameter(torch.full((num_layers,), 0.01))
        self.a_res = nn.Parameter(torch.full((num_layers,), 0.01))
        lane = torch.arange(lanes).float()
        self.register_buffer("_lane", lane)

    def route(self, u, y, layer_idx):
        """u: (B,T,d_model) mixed input to block; y: (B,T,d_model) block delta;
        x prev lanes (B,T,L,d_model). Returns new (B,T,L,d_model)."""
        raise NotImplementedError


class SimpleAttentionNetwork(nn.Module):
    def __init__(self, cfg: SANConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = getattr(torch, cfg.dtype)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=self.dtype)
        nn.init.normal_(self.embedding.weight, 0, _STD)
        self.embed_scale = math.sqrt(cfg.d_model)

        orders, heads, sub_dim = engram_geometry(cfg)
        self.engrams = nn.ModuleList([
            Engram(cfg.d_model, len(orders) * heads, cfg.engram_slots, sub_dim,
                   cfg.num_layers, max(orders), self.dtype)
            for _ in cfg.engram_layers
        ])
        self.blocks = nn.ModuleList([
            Block(cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.num_layers,
                  self.dtype, attn_dim=cfg.attn_dim)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = ZCRMSNorm(cfg.d_model, dtype=self.dtype)

        # MTP (multi-token prediction) head — faithful port: emb_norm, combine,
        # one extra block, final_norm.
        self.mtp_emb_norm = ZCRMSNorm(cfg.d_model, dtype=self.dtype)
        self.mtp_combine = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False, dtype=self.dtype)
        nn.init.normal_(self.mtp_combine.weight, 0, _STD)
        self.mtp_block = Block(cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.num_layers,
                               self.dtype, attn_dim=cfg.attn_dim)
        self.mtp_final_norm = ZCRMSNorm(cfg.d_model, dtype=self.dtype)

        # Auxiliary heads (tool retrieval/confidence) — faithful port, used by
        # the needle agent stack; optional at training time.
        self.contrastive_head = ContrastiveHead(cfg.d_model, 128, self.dtype)
        self.confidence_head = ConfidenceHead(cfg.d_model, self.dtype)

        # Multi-lane params (MHC): per-layer projection matrices
        lanes = cfg.mhc_lanes
        nC = lanes * cfg.d_model
        self.phi_pre = nn.Parameter(torch.empty(cfg.num_layers, nC, lanes).normal_(0, _STD))
        self.phi_post = nn.Parameter(torch.empty(cfg.num_layers, nC, lanes).normal_(0, _STD))
        self.phi_res = nn.Parameter(torch.empty(cfg.num_layers, nC, lanes * lanes).normal_(0, _STD))
        self.b_pre = nn.Parameter(torch.zeros(cfg.num_layers, lanes))
        self.b_post = nn.Parameter(torch.zeros(cfg.num_layers, lanes))
        self.b_res = nn.Parameter(torch.zeros(cfg.num_layers, lanes, lanes))
        with torch.no_grad():
            for L in range(cfg.num_layers):
                self.b_res[L] = 4.0 * torch.eye(lanes)
        self.a_pre = nn.Parameter(torch.full((cfg.num_layers,), 0.01))
        self.a_post = nn.Parameter(torch.full((cfg.num_layers,), 0.01))
        self.a_res = nn.Parameter(torch.full((cfg.num_layers,), 0.01))
        lane_vals = torch.arange(lanes).float()
        self.register_buffer("_pre_off", _LANE_OFF * lane_vals - 4.0)
        self.register_buffer("_post_off", _POST_OFF * (1 - lane_vals))

        # buffers to avoid CUDA graph recompiles per seq len
        self._rope_cache = {}

    def _rope(self, seq_len):
        hd = (self.cfg.attn_dim or self.cfg.d_model) // self.cfg.num_heads
        dev = next(self.parameters()).device
        if seq_len not in self._rope_cache:
            cos, sin = precompute_rope_freqs(hd, seq_len, self.cfg.rope_theta)
            self._rope_cache[seq_len] = (cos.to(dev), sin.to(dev))
        return self._rope_cache[seq_len]

    def _engram_kv(self, tokens, mask):
        cfg = self.cfg
        orders, heads, _ = engram_geometry(cfg)
        indices = engram_indices(tokens, orders, heads, cfg.engram_slots)
        ngram_ok = torch.stack([_mask_diag(mask, o - 1) for o in orders for _ in range(heads)],
                               dim=-1)
        tap_ok = torch.stack([_mask_diag(mask, j * max(orders)) for j in range(ENGRAM_CONV_TAPS)])
        pairs = [e(indices, ngram_ok, tap_ok) for e in self.engrams]
        return torch.stack([k for k, _ in pairs]), torch.stack([v for _, v in pairs])

    def forward(self, tokens, mask=None, return_mtp=False):
        cfg = self.cfg
        if mask is None:
            mask = make_causal_mask(tokens.shape[1], tokens.device)
        B, T = tokens.shape
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(T)
        engram_kv = self._engram_kv(tokens, mask)

        # site flags: which layer fires each engram site
        site_flags = torch.zeros((cfg.num_layers, len(cfg.engram_layers)),
                                 device=tokens.device)
        for s, layer in enumerate(cfg.engram_layers):
            site_flags[layer, s] = 1.0

        # Multi-lane: broadcast to L lanes. fs = 4*d_model
        lanes = cfg.mhc_lanes
        nC = lanes * cfg.d_model
        X = x.unsqueeze(2).expand(B, T, lanes, cfg.d_model).contiguous()  # (B,T,L,d)
        lane_i = torch.arange(cfg.num_layers, device=tokens.device) % lanes

        for i in range(cfg.num_layers):
            xf = X.float()
            nx = _rms_unit(X.reshape(B, T, nC))
            hpre = torch.sigmoid(
                self.a_pre[i] * (nx @ self.phi_pre[i].float()) + self.b_pre[i] + self._pre_off)
            u = torch.einsum("btn,btnc->btc", hpre, xf).to(self.dtype)  # (B,T,d)

            # engram site injection: full site stack + per-layer site_flags row
            block_in = self.blocks[i](u, mask=mask, rope=rope,
                                      engram_kv=engram_kv if engram_kv is not None else None,
                                      site_flags=site_flags[i])
            y = block_in - u  # block delta (B,T,d)

            hpost = 2 * torch.sigmoid(
                self.a_post[i] * (nx @ self.phi_post[i].float()) + self.b_post[i] + self._post_off)
            res = nx @ self.phi_res[i].float()  # (B,T,L*L)
            hres = _sinkhorn(self.a_res[i] * res.reshape(B, T, lanes, lanes) + self.b_res[i])
            new_x = (torch.einsum("btij,btjc->btic", hres, xf)
                     + hpost.unsqueeze(-1) * y.unsqueeze(2).float()).to(self.dtype)
            X = new_x

        x = X.mean(dim=2)
        x = self.final_norm(x)
        logits = x.to(self.dtype) @ self.embedding.weight.T
        if not return_mtp:
            return logits
        # MTP: predict token t+2. Combine main output x with embedding of token
        # t+1, run one more block, then tied-head logits (faithful to JAX).
        rope = self._rope(T)
        nxt = F.pad(tokens[:, 1:], (0, 1))
        e2 = self.mtp_emb_norm(self.embedding(nxt) * self.embed_scale)
        m = self.mtp_combine(torch.cat([x, e2], dim=-1).to(self.dtype))
        m = self.mtp_block(m, mask=mask, rope=rope)
        m = self.mtp_final_norm(m)
        mtp_logits = m.to(self.dtype) @ self.embedding.weight.T
        return logits, mtp_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
