# x-small (135M) — Full Pipeline Abstraction

> Maps the complete lifecycle of the x-small project: two parallel model tracks
> that share one SmolLM2-tokenized shard corpus. Repo: `git@github.com:kenpeter/x-small.git`
> (SSH, `main`). Symmetric sibling of `work/small/abstraction.md` (the 1B project).

---

## Track A: SmolLM2-135M pretrain → SFT  (original pipeline)

A from-scratch Transformer++ pretrain + SFT on the shared shard corpus, sized to
fit a single RTX 4070 Ti 12GB with no CPU offload. This was the original x-small
pipeline before the SAN track was added; it remains intact and unaffected by the
SAN port.

### Architecture — SmolLM2-135M (`model.py`)

Transformer++ (RMSNorm, SwiGLU, RoPE, GQA):

| Spec | Value |
|------|-------|
| Total parameters | ~135M |
| Hidden dim (`dim`) | 576 |
| Layers (`n_layers`) | 30 |
| Attention heads | 9 |
| KV heads | 3 (GQA 3:1) |
| FFN intermediate | 1536 (SwiGLU) |
| Max seq len | 2048 |
| RoPE θ | 10,000 |
| Vocab | 49152 (SmolLM2 tokenizer) |
| Weight tying | No (lm_head untied) |

### Training config (`train.py`)

| Spec | Value |
|------|-------|
| batch × grad-accum | 4 × 8 = **eff 32** (maxed for 12GB; fwd+bwd peak ~10.65GB with fused CE, batch 6+ OOMs) |
| Gradient checkpointing | **On** (enabled 2026-08 — drops VRAM 10.9GB→8.5GB; without it batch 4×8 OOMs) |
| LR | 4e-4, cosine → min 1e-4, warmup 2000 |
| weight_decay | 0.1, AdamW (β 0.9/0.95, eps 1e-8) |
| max_steps | 100,000 |
| loss | fused chunked CE (no full [B,S,V] fp32 logits materialized) |
| TF32 | enabled (negligible accuracy loss on Ampere+) |
| save_every / log / val | 2000 / 10 / 500 |

**Data tricks (git log notes):**
- **Paired reversal per shard** (`d461d3b`, arXiv:2604.00260) — odd shards iterate reversed to cancel order bias.
- **Latest-only checkpoints** — `a3a81fc` persists true best_loss and stops best.pt clobber on resume; `8c58452` appends to training log (`tee -a`) so trajectory survives restarts.
- **Cron/background safety** (`531781a`) — unset poison env vars in training wrapper.

### SFT pipeline
`tokenize_sft.py` / `download_sft.py` take instruct data → `.bin` shards;
`sft.py` fine-tunes from the pretrain checkpoint. Evals: `eval_quick.py`
(quick greedy gen on latest), `eval_sft.py` (SFT checkpoint), `eval_code.py`
(coding-focused greedy gen).

---

## Track B: Simple Attention Network (SAN) — needle port  ⭐ current

A **faithful PyTorch port** of the JAX/Flax `cactus-compute/needle` architecture
(arXiv:2607.18363), created 2026-08-21. Chosen over JAX so it runs in the same
torch pipeline / venv / data-mix as Track A. User directive: *"port as much as
can"* — full SAN, not a simplified subset.

### Model: `san_model.py`

Components, 1:1 with needle `model/architecture.py`:

| Component | Description |
|-----------|-------------|
| `ZCRMSNorm` | Zero-centered RMSNorm |
| `HadamardMLP` | Fixed Walsh-Hadamard transform MLP — **no dense FFN weights** (3 learned diag vectors) |
| `MultiHeadAttention` | GQA + q/k RMSNorm + RoPE + learned gate |
| `Engram` | Hashed n-gram KV memory tables + conv taps, injected at selected sites |
| `Multi-Lane Hyper-Connections` (MHC) | Sinkhorn routing across 4 residual lanes (phi_pre/post/res, bias terms) |
| `MTP` | Multi-token-prediction aux head |
| Heads | Tied embedding + causal/packing mask; contrastive/confidence heads |
| Masks | `make_causal_mask`, `make_causal_packing_mask` (packing + sliding window) |

### Config — the needle2 real preset (NOT `PRESETS["needle"]`)

Inspecting the downloaded `needle2.pkl` (90.4MB, `~/work/needle2.pkl`) showed it
uses **`PRESETS["base"]`** (d_model 512), which is why earlier 87M-vs-45M
estimates were off. Real config:

| Spec | Value |
|------|-------|
| vocab_size | 8192 (native needle) / **49152 (x-small SmolLM2)** |
| d_model | **384** (shrunk from 512 to hit ~45M at 49k vocab) |
| num_heads | 8 |
| num_kv_heads | 4 |
| num_layers | 27 |
| max_seq_len | 2048 |
| engram_layers | (2, 15) — engram_heads 2, orders (2,3), sub_dim 128, tables 4, slots 8192 |
| MHC | 4 lanes (phi 27×2048×4, res 27×2048×16, b 27×4 / 27×4×4) |
| rope_theta | 1e5 |

**Param counts (verified):**

| Vocab / d_model | Total | Embedding | Non-embedding |
|-----------------|-------|-----------|---------------|
| 8192 × 512 (needle2-exact) | **45.21M** | 4.19M | 41.02M ✓ matches needle2.pkl |
| 49152 × 512 (old Option B) | 66.18M | 25.17M | 41.02M |
| **49152 × 384 (CURRENT ~45M)** | **43.85M** | 18.87M | 24.98M |

**Pivot (user directive 2026-08-21):** "I do not want 135m any more. I want
45m like needle or similar small sizes." → d_model 384, 27 layers, vocab 49152
= **43.85M**, ~needle2 scale, keeps SmolLM2 vocab (no re-tokenization). Verified
forward+MTP+backward on GPU, ~3GB VRAM @ seq 512 (fits 12GB at full 2048×b32).

### Training: `train_san.py`

- Reuses `BinShardDataset` over shared shard corpus + latest-only checkpointing.
- `san_loss()` = main next-token CE (+ optional MTP aux via `--mtp-weight`, default 0.1).
- Fixed issues during bring-up:
  - **RoPE device bug** — `precompute_rope_freqs` built tables on CPU; now moved via `._rope()` to model device.
  - **MTP loss alignment** — main CE compares full `main` vs `y` (no `[:, :-1]` slice); MTP (`mtp[:, :-1]` vs `y[:, 1:]`) predicts token[t+2] from x[t]+emb(tok[t+1]).
- Verified **running on GPU (cuda:0, RTX 4070 Ti)**: loss 11.97→11.91→11.70 over 30 steps, ~2.5–3.0k tok/s @ batch 1 / seq 512.
- Config defaults: batch 4 × accum 8 = eff 32, seq 2048, lr 4e-4 cos→1e-4, warmup 1000, save_every 2000, val_frac 0.01, checkpointing to `/home/kenpeter/work/checkpoints/xsmall_san` (`san_latest.pt`).

### Commits
- `cfa5334` — port SAN to PyTorch (45.21M params, faithful). *(device + loss fixes after this are NOT yet committed — see Pending.)*

---

## Shared Data Corpus

Same SmolLM2-135M-tokenized `.bin` uint16 shard corpus as the 1B project:

| Location | Contents |
|----------|----------|
| `/home/kenpeter/work/data/_shards_final` | ~1098 shards, SmolLM2 vocab 49152 (flat farm default) |
| Tiered domain shards | easy/medium/hard per domain (math/web/synth/code + gold), symlinked for curriculum use |
| Shard format | `.bin` uint16 arrays, seq chunks of `seq_len+1` |

`train_san.py` supports flat farm by default; `--curriculum` (stratified
domain-tiered sampler ported from small) is stubbed/planned but not yet wired.

---

## Environment & Conventions

- **venv**: `venv_xsmall/` (Python 3.11). System python lacks some deps — always use `./venv_xsmall/bin/python`.
- **Device**: RTX 4070 Ti 12GB; `device = "cuda" if torch.cuda.is_available()` everywhere.
- **Checkpointing rule (user)**: NEVER save `best.pt` — always latest, resume from latest. Independent stage-isolated checkpoints (pretrain → sft).
- **Git**: real repo is `work/x-small` → GitHub `kenpeter/x-small` (SSH). Commit after code changes.
- **Deletion rule (user)**: never delete files without exploring + explicit approval.

---

## Status & Pending

✅ Done: SAN port (45.21M @ needle2-exact / **43.85M @ 49152 current**),
~45M scale locked (d_model 384, 27L — user rejected 135M/66M).

✅ GPU optimization (`ad2b800`): **Triton Sinkhorn** (1.13→0.095ms, 12×, parity
2.4e-7), **flash SDPA attention** (kills the (B,H,T,T) fp32 OOM), **grad
checkpointing** (use_checkpoint — b4×T2048 fwd+bwd fits 5.9GB, was OOM). Full
train smoke OK: 43.85M, loss 23.89@step10, ~9k tok/s. FWHT kernel deferred
(Triton static_range constexpr limits; dense cuBLAS GEMM kept at n=512).

✅ **DoReMi-lite curriculum ported** (`e920290`): `--curriculum` flag →
StratifiedShardDataset over 16 tiered domains, ratio-driven interleave + JIT
shuffle; `get_curriculum_ratios()` (G1 boundary sharpening + G2 cyclic review +
G3 smooth glide) re-glided every 2000 steps via `ds.reweight()` (hot-swap, no
restart); `curriculum_boost.json` hot-reload with code-dominant boost
(code_easy 21×, code_medium 12×, code_hard 21×, code_gold 10×) for the exam-style
code-pass goal. Verified: 16 sources, resume-from-latest, 34k tok/s.

🔜 Pending:
1. **Launch real GPU run** — curriculum `--curriculum`, batch 4 × accum 8, seq 2048, 50k steps, detached + watchdog.
2. **Eval** trained SAN on code/math prompts.
3. Housekeeping: `token_rotation_test.py` committed with tokens redacted (HF tokens removed — were in git, push-protection flagged).
