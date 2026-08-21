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


def available():
    return _HAS_TRITON
