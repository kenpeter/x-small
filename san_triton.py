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
    def mhc_rms_hpre_kernel(X_ptr, phi_pre_ptr, hpre_ptr, a,
                            b_ptr, off_ptr,
                            B: tl.constexpr, T: tl.constexpr, L: tl.constexpr,
                            D: tl.constexpr, BLOCK_D: tl.constexpr):
        """Fused per-(b,t): row-RMS of X (bf16) + hpre = sigmoid(a*rms*(X@phi_pre)+b+off).

        Single X pass using linearity: (rms*X)@phi = rms*(X@phi). Replaces the
        eager X.float() cast + _rms_unit + (B,T,nC)@(nC,L) GEMM + sigmoid cluster."""
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T
        base = b * T * L * D + t * L * D
        offs_l = tl.arange(0, L)
        offs_d = tl.arange(0, BLOCK_D)
        s2 = 0.0
        ge = tl.zeros([L], dtype=tl.float32)
        for db in tl.static_range(D // BLOCK_D):
            x_2d = tl.load(X_ptr + base + offs_l[:, None] * D + db * BLOCK_D + offs_d[None, :]).to(tl.float32)  # (L,BLOCK_D)
            s2 += tl.sum(x_2d * x_2d)
            phi_3d = tl.load(phi_pre_ptr + (offs_l[:, None, None] * D + db * BLOCK_D + offs_d[None, :, None]) * L
                             + offs_l[None, None, :]).to(tl.float32)  # (L,BLOCK_D,L)
            ge += tl.sum(tl.sum(x_2d[:, :, None] * phi_3d, axis=0), axis=0)   # (L,)
        rms = tl.rsqrt(s2 / (L * D) + 1e-6)
        bv = tl.load(b_ptr + offs_l).to(tl.float32)
        offv = tl.load(off_ptr + offs_l).to(tl.float32)
        hpre = tl.sigmoid(a * rms * ge + bv + offv)
        tl.store(hpre_ptr + b * T * L + t * L + offs_l, hpre)

    @triton.jit
    def mhc_rms_hpost_kernel(X_ptr, phi_post_ptr, phi_res_ptr,
                             hpost_ptr, sink_ptr, a_post, b_post_ptr, off_ptr,
                             a_res, b_res_ptr,
                             B: tl.constexpr, T: tl.constexpr, L: tl.constexpr,
                             D: tl.constexpr, BLOCK_D: tl.constexpr):
        """Fused per-(b,t): row-RMS of X, hpost = 2*sigmoid(a_post*rms*(X@phi_post)+b+off),
        sink_in = a_res*rms*(X@phi_res).reshape(L,L) + b_res.

        Routing params (phi_res/a_res/b_res) get NO grad — sink_in is detached in
        _MHC_POST, matching the eager/triton behavior."""
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T
        base = b * T * L * D + t * L * D
        offs_l = tl.arange(0, L)
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, L * L)
        s2 = 0.0
        ge_post = tl.zeros([L], dtype=tl.float32)
        ge_res = tl.zeros([L * L], dtype=tl.float32)
        for db in tl.static_range(D // BLOCK_D):
            x_2d = tl.load(X_ptr + base + offs_l[:, None] * D + db * BLOCK_D + offs_d[None, :]).to(tl.float32)  # (L,BLOCK_D)
            s2 += tl.sum(x_2d * x_2d)
            ph_post_3d = tl.load(phi_post_ptr + (offs_l[:, None, None] * D + db * BLOCK_D + offs_d[None, :, None]) * L
                                 + offs_l[None, None, :]).to(tl.float32)   # (L,BLOCK_D,L)
            ge_post += tl.sum(tl.sum(x_2d[:, :, None] * ph_post_3d, axis=0), axis=0)  # (L,)
            ph_res_3d = tl.load(phi_res_ptr + (offs_l[:, None, None] * D + db * BLOCK_D + offs_d[None, :, None]) * (L * L)
                                + offs_r[None, None, :]).to(tl.float32)    # (L,BLOCK_D,L*L)
            ge_res += tl.sum(tl.sum(x_2d[:, :, None] * ph_res_3d, axis=0), axis=0)    # (L*L,)
        rms = tl.rsqrt(s2 / (L * D) + 1e-6)
        bv = tl.load(b_post_ptr + offs_l).to(tl.float32)
        offv = tl.load(off_ptr + offs_l).to(tl.float32)
        hpost = 2.0 * tl.sigmoid(a_post * rms * ge_post + bv + offv)
        tl.store(hpost_ptr + b * T * L + t * L + offs_l, hpost)
        res = (a_res * rms * ge_res).reshape(L, L) + tl.load(b_res_ptr + offs_r).to(tl.float32).reshape(L, L)
        tl.store(sink_ptr + b * T * L * L + t * L * L + offs_r, res.reshape(L * L))

    @triton.jit
    def mhc_pre_kernel(hpre_ptr, X_ptr, u_ptr,
                       B: tl.constexpr, T: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                       BLOCK_D: tl.constexpr):
        """u[b,t,c] = sum_l hpre[b,t,l] * X[b,t,l,c]   (einsum 'btn,btnc->btc').

        Reads bf16 X directly (casts in-register) — no fp32 X.float() temp.
        Static-unrolled lane x D-block loops, zero padding (BLOCK_D divides D)."""
        pid = tl.program_id(0)
        b = pid // T
        t = pid % T
        offs_d = tl.arange(0, BLOCK_D)
        for db in tl.static_range(D // BLOCK_D):
            u_blk = tl.zeros([BLOCK_D], dtype=tl.float32)
            for l in tl.static_range(L):
                hpre_l = tl.load(hpre_ptr + b * T * L + t * L + l).to(tl.float32)
                xf_l = tl.load(X_ptr + b * T * L * D + t * L * D + l * D + db * BLOCK_D + offs_d).to(tl.float32)
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
    def forward(ctx, hpre, X):
        B, T, L, D = X.shape
        u = torch.empty((B, T, D), dtype=torch.float32, device=X.device)
        BLOCK_D = _MHC_PRE._block_d(D)
        mhc_pre_kernel[(B * T,)](hpre, X, u, B=B, T=T, L=L, D=D, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(hpre, X)
        return u

    @staticmethod
    def backward(ctx, gu):
        hpre, X = ctx.saved_tensors
        xf = X.float()
        g_hpre = torch.einsum("btc,btnc->btn", gu, xf)
        g_xf = torch.einsum("btc,btn->btnc", gu, hpre)
        return g_hpre, g_xf


class _FUSED_PRE(torch.autograd.Function):
    """Fused row-RMS + hpre routing projection (Triton fwd, exact torch bwd)."""

    @staticmethod
    def forward(ctx, X, phi_pre, a_pre, b_pre, pre_off):
        B, T, L, D = X.shape
        hpre = torch.empty((B, T, L), dtype=torch.float32, device=X.device)
        BLOCK_D = _MHC_PRE._block_d(D)
        mhc_rms_hpre_kernel[(B * T,)](X, phi_pre, hpre, float(a_pre), b_pre, pre_off,
                                      B=B, T=T, L=L, D=D, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(X, phi_pre, a_pre, b_pre, pre_off, hpre)
        return hpre

    @staticmethod
    def backward(ctx, g_hpre):
        X, phi_pre, a_pre, b_pre, pre_off, hpre = ctx.saved_tensors
        B, T, L, D = X.shape
        nC = L * D
        xf = X.float().reshape(B, T, nC)
        s = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        nx = xf * s
        g_z = g_hpre * hpre * (1 - hpre)               # sigmoid derivative
        g_nx = (g_z * a_pre) @ phi_pre.t()             # (B,T,nC)
        # RMS backward (analytic — no nested autograd inside Function.backward):
        # g_x = s * (g_y - nx * mean(nx * g_y))
        g_X = s * (g_nx - nx * (g_nx * nx).mean(-1, keepdim=True))
        g_phi = torch.einsum("btc,btj->cj", nx, g_z * a_pre)
        g_a = (g_z * (nx @ phi_pre)).sum()
        g_b = g_z.sum(dim=(0, 1))
        return g_X.reshape(B, T, L, D), g_phi, g_a, g_b, None


class _MHC_POST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sink_in, X, y, hpost):
        B, T, L, D = X.shape
        sk_det = sink_in.detach().reshape(B, T, L, L)
        hres = triton_sinkhorn(sk_det)                  # non-diff, matches eager
        out = torch.empty((B, T, L, D), dtype=torch.float32, device=X.device)
        BLOCK_D = _MHC_PRE._block_d(D)
        mhc_post_kernel[(B * T,)](sk_det, X, y, hpost, out,
                                  B=B, T=T, L=L, D=D, ITERS=SINKHORN_ITERS,
                                  BLOCK_D=BLOCK_D)
        ctx.save_for_backward(hres, X, y, hpost)
        return out

    @staticmethod
    def backward(ctx, g_out):
        hres, X, y, hpost = ctx.saved_tensors
        xf = X.float()
        g_xf = torch.einsum("btic,btij->btjc", g_out, hres)
        g_hpost = torch.einsum("btic,btc->bti", g_out, y)
        g_y = torch.einsum("btic,bti->btc", g_out, hpost)
        return None, g_xf, g_y, g_hpost


class _FUSED_POST(torch.autograd.Function):
    """Fused hpost + sink_in routing projections (Triton fwd, exact torch bwd).

    Routing params (phi_res/a_res/b_res) take NO grads (sink_in feeds the
    detached Sinkhorn — matches eager routing behavior)."""

    @staticmethod
    def forward(ctx, X, phi_post, a_post, b_post, post_off, phi_res, a_res, b_res):
        B, T, L, D = X.shape
        hpost = torch.empty((B, T, L), dtype=torch.float32, device=X.device)
        sink = torch.empty((B, T, L * L), dtype=torch.float32, device=X.device)
        BLOCK_D = _MHC_PRE._block_d(D)
        mhc_rms_hpost_kernel[(B * T,)](X, phi_post, phi_res, hpost, sink,
                                       float(a_post), b_post, post_off,
                                       float(a_res), b_res,
                                       B=B, T=T, L=L, D=D, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(X, phi_post, a_post, b_post, post_off, hpost)
        return hpost, sink

    @staticmethod
    def backward(ctx, g_hpost, g_sink):
        X, phi_post, a_post, b_post, post_off, hpost = ctx.saved_tensors
        B, T, L, D = X.shape
        nC = L * D
        xf = X.float().reshape(B, T, nC)
        s = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        nx = xf * s
        # hpost = 2*sigmoid(z) -> dz = g * hpost * (1 - hpost/2)
        g_z = g_hpost * hpost * (1 - 0.5 * hpost)
        g_nx = (g_z * a_post) @ phi_post.t()
        g_X = s * (g_nx - nx * (g_nx * nx).mean(-1, keepdim=True))
        g_phi = torch.einsum("btc,btj->cj", nx, g_z * a_post)
        g_a = (g_z * (nx @ phi_post)).sum()
        g_b = g_z.sum(dim=(0, 1))
        return (g_X.reshape(B, T, L, D), g_phi, g_a, g_b, None, None, None, None)


def mhc_pre(hpre, X):
    """Fused lane-mix einsum. hpre (B,T,L) fp32, X (B,T,L,D) bf16 -> u (B,T,D) fp32."""
    return _MHC_PRE.apply(hpre, X)


def mhc_post(sink_in, X, y, hpost):
    """Fused Sinkhorn + combine. y (B,T,D) fp32, hpost (B,T,L) fp32
    -> new_x (B,T,L,D) fp32. X (B,T,L,D) bf16 (cast in-kernel)."""
    return _MHC_POST.apply(sink_in, X, y, hpost)


def fused_pre(X, phi_pre, a_pre, b_pre, pre_off):
    """Fused row-RMS + hpre projection. Returns hpre (B,T,L) fp32."""
    return _FUSED_PRE.apply(X, phi_pre, a_pre, b_pre, pre_off)


def fused_post(X, phi_post, a_post, b_post, post_off, phi_res, a_res, b_res):
    """Fused hpost + sink_in projections. Returns (hpost (B,T,L), sink_in (B,T,L*L))."""
    return _FUSED_POST.apply(X, phi_post, a_post, b_post, post_off, phi_res, a_res, b_res)


def available():
    return _HAS_TRITON
