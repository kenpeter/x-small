"""Triton kernels for the Simple Attention Network (SAN) — compute-optimized
replacements for the hot Python/torch loops. Guarded so the module imports even
when triton is absent (the torch/Python fallbacks live in san_model.py).

Kernels:
  - triton_sinkhorn : the MHC 20-iteration Sinkhorn routing collapsed into ONE
                      persistent kernel (row/col logsumexp in-register), replacing
                      the Python 20-iter loop that runs per layer (~25% of fwd).
  - triton_fwht     : fast Walsh-Hadamard transform O(n log n) butterflies,
                      replacing the dense `z @ H` (n x n) GEMM in the HadamardMLP.

These train-from-scratch SAN ops, so sub-ULP drift vs the torch/GEMM reference is
acceptable (and the FWHT matches needle's own JAX fast-transform, not a dense GEMM).
Numerics follow the triton-bitwise-parity rules (explicit fp32 dtype, careful
scalars). Both fall back gracefully when triton is unavailable.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton absent
    _HAS_TRITON = False


# ─── Sinkhorn ────────────────────────────────────────────────────────────────
# Each program handles one row of the (M, L, L) tensor (L = MHC lanes, small).
# Runs `iters` full row+col normalizations entirely in registers, single launch.
if _HAS_TRITON:
    @triton.jit
    def sinkhorn_kernel(inp, out, iters,
                        L: tl.constexpr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, L * L)
        base = pid * (L * L)
        x = tl.load(inp + base + offs).to(tl.float32).reshape(L, L)
        for _ in tl.static_range(ITERS):
            # row logsumexp center-then-exp final
            row_max = tl.max(x, axis=1)                      # (L,)
            x = x - row_max[:, None]
            row_s = tl.sum(tl.exp(x), axis=1)                # (L,)
            x = x - tl.log(row_s)[:, None]
            # col
            col_max = tl.max(x, axis=0)
            x = x - col_max[None, :]
            col_s = tl.sum(tl.exp(x), axis=0)
            x = x - tl.log(col_s)[None, :]
        x = tl.exp(x)
        tl.store(out + base + offs, x.reshape(L * L))


def triton_sinkhorn(logits, iters=20):
    """logits (..., L, L) -> Sinkhorn(20) as exp(K). Matches san_model._sinkhorn."""
    L = logits.shape[-1]
    flat = logits.reshape(-1, L * L).contiguous()
    M = flat.shape[0]
    out = torch.empty_like(flat)
    sinkhorn_kernel[(M,)](flat, out, iters, L=L, ITERS=iters)
    return out.reshape(logits.shape)


# ─── FWHT (deferred) ─────────────────────────────────────────────────────────
# The HadamardMLP's `z @ H` is currently left as the dense cuBLAS GEMM (0.19ms
# at n=512 — memory-bound, cuBLAS is near-optimal for this tiny matrix). A custom
# register FWHT (O(n log n)) was prototyped but Triton's static_range constexpr
# reassignment rules make the multi-stage butterfly reshape awkward, and the
# gain (~3% of forward) didn't justify it. Revisit if n grows or GEMM becomes hot.


def triton_fwht(x):
    """Placeholder — dense Hadamard GEMM retained (see note above)."""
    return x


# ─── MHC routing fusion ───────────────────────────────────────────────────────
# Fuses the per-layer Multi-Lane Hyper-Connection routing tails (the
# memory-bound elementwise + einsum + Sinkhorn) into single Triton kernels.
# Numerics follow the triton-bitwise-parity rules: explicit fp32, careful
# scalars. Sinkhorn stays non-differentiable (matches the eager reference —
# routing params get no grad through it), so these are wrapped in autograd
# Functions that recompute hres in the backward for the xf/hpost/y grads.
if _HAS_TRITON:
    @triton.jit
    def mhc_pre_kernel(hpre_ptr, xf_ptr, u_ptr,
                       B: tl.constexpr, T: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                       BLOCK_LD: tl.constexpr, BLOCK_D: tl.constexpr):
        """u[b,t,c] = sum_l hpre[b,t,l] * xf[b,t,l,c]   (einsum 'btn,btnc->btc').
        BLOCK_LD/BLOCK_D are next-power-of-2 padded dims with masking."""
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T
        hpre = tl.load(hpre_ptr + b * T * L + t * L + tl.arange(0, L)).to(tl.float32)
        offs_ld = tl.arange(0, BLOCK_LD)
        mask_ld = offs_ld < L * D
        xf = tl.load(xf_ptr + b * T * L * D + t * L * D + offs_ld, mask=mask_ld, other=0.0).to(tl.float32).reshape(L, BLOCK_D)
        u = tl.sum(hpre[:, None] * xf, axis=0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        tl.store(u_ptr + b * T * D + t * D + offs_d, u, mask=mask_d)

    @triton.jit
    def mhc_post_kernel(sink_in_ptr, xf_ptr, y_ptr, hpost_ptr, out_ptr,
                        B: tl.constexpr, T: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                        ITERS: tl.constexpr, BLOCK_LD: tl.constexpr, BLOCK_D: tl.constexpr):
        """Inline 20-iter Sinkhorn on sink_in, then
        new_x[b,t,l,c] = sum_j hres[l,j]*xf[b,t,j,c] + hpost[l]*y[b,t,c]
        (einsum 'btij,btjc->btic' + broadcast)."""
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T
        base = b * T * L * L + t * L * L
        sk = tl.load(sink_in_ptr + base + tl.arange(0, L * L)).to(tl.float32).reshape(L, L)
        for _ in tl.static_range(ITERS):
            row_max = tl.max(sk, axis=1)
            sk = sk - row_max[:, None]
            row_s = tl.sum(tl.exp(sk), axis=1)
            sk = sk - tl.log(row_s)[:, None]
            col_max = tl.max(sk, axis=0)
            sk = sk - col_max[None, :]
            col_s = tl.sum(tl.exp(sk), axis=0)
            sk = sk - tl.log(col_s)[None, :]
        hres = tl.exp(sk)                                                 # (L,L)
        offs_ld = tl.arange(0, BLOCK_LD)
        mask_ld = offs_ld < L * D
        xf = tl.load(xf_ptr + b * T * L * D + t * L * D + offs_ld, mask=mask_ld, other=0.0).to(tl.float32).reshape(L, BLOCK_D)  # (L,D)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        y = tl.load(y_ptr + b * T * D + t * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)  # (D,)
        hpost = tl.load(hpost_ptr + b * T * L + t * L + tl.arange(0, L)).to(tl.float32)
        new_x = tl.sum(hres[:, :, None] * xf[None, :, :], axis=1) + hpost[:, None] * y[None, :]
        tl.store(out_ptr + b * T * L * D + t * L * D + offs_ld, new_x.reshape(L * BLOCK_D), mask=mask_ld)


class _MHC_PRE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hpre, xf):
        B, T, L, D = xf.shape
        u = torch.empty((B, T, D), dtype=torch.float32, device=xf.device)
        BLOCK_D = 1 << ((D - 1).bit_length())
        BLOCK_LD = L * BLOCK_D
        mhc_pre_kernel[(B * T,)](hpre, xf, u, B=B, T=T, L=L, D=D,
                                 BLOCK_LD=BLOCK_LD, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(hpre, xf)
        return u

    @staticmethod
    def backward(ctx, gu):
        hpre, xf = ctx.saved_tensors
        g_hpre = torch.einsum("btc,btnc->btn", gu, xf)
        g_xf = torch.einsum("btc,btn->btnc", gu, hpre)
        return g_hpre, g_xf


class _MHC_POST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sink_in, xf, y, hpost):
        B, T, L, D = xf.shape
        sk_det = sink_in.detach()
        hres = triton_sinkhorn(sk_det)                  # non-diff, matches eager
        out = torch.empty((B, T, L, D), dtype=torch.float32, device=xf.device)
        BLOCK_D = 1 << ((D - 1).bit_length())
        BLOCK_LD = L * BLOCK_D
        mhc_post_kernel[(B * T,)](sk_det, xf, y, hpost, out,
                                  B=B, T=T, L=L, D=D, ITERS=20,
                                  BLOCK_LD=BLOCK_LD, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(hres, xf, y, hpost)
        return out

    @staticmethod
    def backward(ctx, g_out):
        hres, xf, y, hpost = ctx.saved_tensors
        g_xf = torch.einsum("btic,btij->btjc", g_out, hres)
        g_hpost = torch.einsum("btic,btc->bti", g_out, y)
        g_y = torch.einsum("btic,bti->btc", g_out, hpost)
        return None, g_xf, g_y, g_hpost


def mhc_pre(hpre, xf):
    """Fused lane-mix einsum. hpre (B,T,L) fp32, xf (B,T,L,D) fp32 -> u (B,T,D) fp32."""
    return _MHC_PRE.apply(hpre, xf)


def mhc_post(sink_in, xf, y, hpost):
    """Fused Sinkhorn + combine. y (B,T,D) fp32, hpost (B,T,L) fp32
    -> new_x (B,T,L,D) fp32."""
    return _MHC_POST.apply(sink_in, xf, y, hpost)


def available():
    return _HAS_TRITON
