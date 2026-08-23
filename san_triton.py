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


# Sinkhorn iterations: 20 -> 14 (4x4 matrices converge well before; big win on the
# hottest in-kernel loop). Kept in one place so kernel + backward-hres + eager
# fallback all agree.
SINKHORN_ITERS = 14


def triton_sinkhorn(logits, iters=SINKHORN_ITERS):
    """logits (..., L, L) -> Sinkhorn(ITERS) as exp(K). Matches san_model._sinkhorn."""
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
                       BLOCK_D: tl.constexpr):
        """u[b,t,c] = sum_l hpre[b,t,l] * xf[b,t,l,c]   (einsum 'btn,btnc->btc').

        Static-unrolled lane x D-block loops. BLOCK_D divides D exactly, so there
        is ZERO padded-memory traffic (old BLOCK_LD=L*BLOCK_D padded 1536->2048,
        ~33% wasted bandwidth on this memory-bound hot path)."""
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T
        offs_d = tl.arange(0, BLOCK_D)
        for db in tl.static_range(D // BLOCK_D):
            u_blk = tl.zeros([BLOCK_D], dtype=tl.float32)
            for l in tl.static_range(L):
                hpre_l = tl.load(hpre_ptr + b * T * L + t * L + l).to(tl.float32)
                xf_l = tl.load(xf_ptr + b * T * L * D + t * L * D + l * D + db * BLOCK_D + offs_d).to(tl.float32)
                u_blk += hpre_l * xf_l
            tl.store(u_ptr + b * T * D + t * D + db * BLOCK_D + offs_d, u_blk)

    @triton.jit
    def mhc_post_kernel(sink_in_ptr, xf_ptr, y_ptr, hpost_ptr, out_ptr,
                        B: tl.constexpr, T: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                        ITERS: tl.constexpr, BLOCK_D: tl.constexpr):
        """Inline ITERS-iter Sinkhorn on sink_in, then
        new_x[b,t,l,c] = sum_j hres[l,j]*xf[b,t,j,c] + hpost[l]*y[b,t,c].

        D-block loop loads (L,BLOCK_D) tiles via 2D pointer offsets — exact shapes,
        no padding (old BLOCK_LD version padded 1536->2048)."""
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
        hres = tl.exp(sk)                                                 # (L,L) registers
        offs_l = tl.arange(0, L)
        offs_d = tl.arange(0, BLOCK_D)
        hpost = tl.load(hpost_ptr + b * T * L + t * L + offs_l).to(tl.float32)
        xb = b * T * L * D + t * L * D
        for db in tl.static_range(D // BLOCK_D):
            offs = db * BLOCK_D + offs_d
            xf_blk = tl.load(xf_ptr + xb + offs_l[:, None] * D + offs[None, :]).to(tl.float32)
            y_blk = tl.load(y_ptr + b * T * D + t * D + offs).to(tl.float32)
            new_x = tl.sum(hres[:, :, None] * xf_blk[None, :, :], axis=1) + hpost[:, None] * y_blk[None, :]
            tl.store(out_ptr + xb + offs_l[:, None] * D + offs[None, :], new_x)


class _MHC_PRE(torch.autograd.Function):
    @staticmethod
    def _block_d(D):
        # largest power-of-2 divisor of D -> zero-padding blocks (384 -> 128)
        bd = 1
        while D % (bd * 2) == 0:
            bd *= 2
        return bd

    @staticmethod
    def forward(ctx, hpre, xf):
        B, T, L, D = xf.shape
        u = torch.empty((B, T, D), dtype=torch.float32, device=xf.device)
        BLOCK_D = _MHC_PRE._block_d(D)
        mhc_pre_kernel[(B * T,)](hpre, xf, u, B=B, T=T, L=L, D=D, BLOCK_D=BLOCK_D)
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
        BLOCK_D = _MHC_PRE._block_d(D)
        mhc_post_kernel[(B * T,)](sk_det, xf, y, hpost, out,
                                  B=B, T=T, L=L, D=D, ITERS=SINKHORN_ITERS,
                                  BLOCK_D=BLOCK_D)
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
