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

### Simple text-flow diagram (forward pass)

```
tokens (B,T)
   │
   ▼
Embedding (tied, 49152×384, × embed_scale)        ── x (B,T,384)
   │
   ▼
Broadcast → 4 MHC lanes                           ── X (B,T,4,384)
   │
   ▼  ┌─ N × MHC layer (i = 0..N-1)  — N=27 default, **18 active** ──┐
   │  │                                                        │
   │  xf = X .float()                                          │
   │  nx = RMS(X) over all lanes        (B,T,1536)             │
   │  hpre = σ(a_pre·(nx@φ_pre) + b_pre)   [lane-mix gate]     │
   │  u = Σ_l hpre[l]·xf[l]              (B,T,384)  ◄─ mhc_pre  │
   │  │                                                       │
   │  u ── Block[i]:                                           │
   │      │  + Engram KV read        (fires only at L=2,15)    │
   │      │  ZCRMSNorm → MHA (GQA+RoPE) → gate                 │
   │      │  ZCRMSNorm → HadamardMLP (no dense FFN weights)    │
   │      └─ y = block_out − u        (delta, B,T,384)         │
   │  │                                                       │
   │  hpost = 2σ(a_post·(nx@φ_post)+b_post)  [routing gate]    │
   │  res = nx@φ_res → sink_in      (B,T,4,4)                  │
   │  hres = Sinkhorn(sink_in, 20)                             │
   │  new_x = hres·xf + hpost·y     ◄─ mhc_post (Triton)       │
   │  │                                                       │
   └── X = new_x ──────────────────────────────────────────────┘
   │
   ▼
X.mean(lanes) → x (B,T,384)
   │
   ▼
FinalNorm → tied logits (chunked over T) → main next-token CE
   └─ MTP aux: x + emb(tok[t+1]) → 1 more block → t+2 logits
```

Lanes: 4 parallel residual streams; per layer `hpre` mixes inputs,
`hres` (Sinkhorn) routes outputs — no dense per-lane FFNs.

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
| num_layers | 27 (default; **live run uses 18** via `--num-layers` for throughput) |
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
- Verified **running on GPU (cuda:0, RTX 4070 Ti)**: loss 11.97→11.91→11.70 over 30 steps, ~2.5–3.0k tok/s @ batch 1 / seq 512. Full curriculum run launched at **~11.6k tok/s** (eager), batch 4 × accum 8, seq 2048 — loss ~2.9 @ step 50260.
- Config defaults: batch 4 × accum 8 = eff 32, seq 2048, lr 4e-4 cos→1e-4, warmup 1000, save_every 2000, val_frac 0.01, **checkpoint-dir `/home/kenpeter/work/x-small/checkpoints/san`** (`san_latest.pt`, latest-only — never best.pt). Launch flags: `--curriculum --dedup` (stratified domain-tiered sampler + dedupe). Runs **with** per-layer `torch.compile` (mode=`default`, checkpoint-preserving); do **not** pass `--no-compile` — eager leaves ~30% throughput on the table (see Throughput section below).
- **Compile-mask graph break (`san_model.py`)** — `MultiHeadAttention.forward` compared masks via `torch.equal(mask, causal_ref)` (data-dependent → torch.compile graph-break + crash when the curriculum mask changes per batch). Fixed: identity check `mask is causal_ref` against a **cached causal mask** (`_causal_mask_cache` dict + `_get_causal_mask()`); `_hidden()` and `compute_loss()` now pull the cached mask from `blocks[0].attn` / `mtp_block.attn` instead of `make_causal_mask(...)`. Compile-safe in principle.
- **torch.compile HANGS (autotune tax pathological)** — with the mask fix, `torch.compile` still never finishes inductor autotune (>7 min wall, 0% GPU, no `✅ compiled compute_loss` line) on this 27-layer + MTP + engram + Triton-Sinkhorn model. Compiled mode was ~15k tok/s (vs ~11.6k eager) but is **unreachable**. Decision at the time: run **`--no-compile` (eager)** — but see the stale-note below: compiling *per-layer* (`_compile_model`, mode=`default`) actually works and is the operational mode; the whole-`compute_loss` compile was the part that hung.

> **⚠️ Stale note — corrected 2026-08-26:** the per-layer `torch.compile` path
> (`_compile_model`, mode=`default`) **does** work and is the operational mode.
> The earlier "compile hangs" was the *whole-`compute_loss`* compile; compiling
> each block forward (checkpoint-preserving) gives 0 recompiles and the 15k+
> result below. Do **not** run `--no-compile` for production.

### Throughput — reaching 16k+ tok/s (2026-08-26)

**Myth busted:** the 15k number was *not* a torch-version artifact, and CUDA
graphs / `max-autotune` do **not** help here. The model is 43.85M params in
**native bf16** at B=3 — it is **overhead-bound, not compute-bound**. PyTorch
issue #171672 confirms `max-autotune`/`reduce-overhead` (CUDA graphs) is *slower*
for this exact shape, and our own measurement showed graphs gave 12.2k vs 12.6k
(no-graphs). A torch 2.11→older **revert was evaluated and proved unnecessary**.

**Real bottleneck = ~90 forced GPU→CPU syncs/step from `.item()` in the hot loop:**
- `chunked_ce()` called `.item()` **8× per microbatch** (once per 256-token chunk).
- the train loop called `.item()` **3–4× per microbatch** (loss / main / mtp / domain).
Each `.item()` drains the pipeline on a model whose microbatch is only a few ms —
a large tax for a tiny model.

**Three changes to `train_san.py` (reversible, git-tracked):**
1. **Killed the `.item()` sync storm** — accumulate *detached* tensors
   (`loss_log`, `acc_main`, `acc_mtp`, `acc_dom_loss`); call `.item()` *only* at
   log intervals. `chunked_ce` keeps `count` as a tensor and returns
   `total / count.clamp(min=1)`.
2. **`x.to(device, non_blocking=True)`** (was blocking) — overlaps H2D with compute.
3. **`persistent_workers=True`** in both DataLoaders (was `False`) — stops the
   8-worker respawn, especially on every curriculum re-glide.

**Results (RTX 4070 Ti 12GB, B=3 + partial grad-ckpt, no OOM):**
| Mode | tok/s |
|------|-------|
| Curriculum + DoReMi, 27L/B=3 (baseline) | ~12.6k (uncapped) / ~11.9k (100W cap) |
| Curriculum + DoReMi, **18L/B=4 (current)** | **~16,700** @100W cap, 8134 MiB |
| Flat farm, 18L/B=4 | higher |

**Power-cap scaling (18L/B=4, 8134 MiB VRAM):** the SM clock is gated by the
power cap, so tok/s scales with it. Measured live:

| Power cap | SM clock | tok/s | Temp | Power draw |
|-----------|----------|-------|------|------------|
| 100W | 1200 MHz | ~16.5k | ~62°C | ~99W |
| 150W | ~2000 MHz (est) | ~19k (est) | ~70°C (est) | ~150W |
| 200W | 2685 MHz | ~22k | ~78°C | ~193W |

Set via `sudo nvidia-smi -pl N` (resets to ~285W default on reboot). 200W =
faster but hotter; 100W = cool.

Both clear 15k with **no torch revert**. Steady-state confirmed live after resume
from `san_latest.pt` (step 52000).

**Operational notes:**
- `--no-checkpoint` is **not** a lever — at B=3 it OOMs at 11.84 GiB on torch 2.11.
grad-ckpt (ckpt_every=1) → 8134 MiB, no OOM. `--num-layers N` (e.g. 18) shrinks the model and is **resume-safe** via `_resume_load` (slices MHC per-layer params to [:N], ignores extra blocks; optimizer reinitialized). 18L + B=4 = ~16,700 tok/s @100W cap, **~22k tok/s @200W cap**.
- Resume cmd: `cd /home/kenpeter/work/x-small && HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES=0 venv_xsmall/bin/python -u train_san.py --resume
  checkpoints/san/san_latest.pt --curriculum --num-layers 18 --batch-size 4 --steps 1000000`.

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
~45M scale locked (d_model 384 — user rejected 135M/66M; `num_layers` default 27, live run uses **18** for throughput).

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

✅ **15k→16.7k tok/s + layer reduction + resume-safe** (2026-08-26): killed the
`.item()` sync storm (see Throughput), then shrank 27→**18 layers** via
`--num-layers 18` — resume-safe through `_resume_load` (slices MHC per-layer
params to [:N], ignores extra blocks; optimizer reinitialized). B=4 forces full
grad-ckpt (ckpt_every=1) → 8134 MiB. Live run resumed from `san_latest.pt` (step
52000) at **~16,700 tok/s @100W cap** (+40% vs 27L/B=3). Engram fusion deferred
(low ROI — Sinkhorn already Triton-fused at `san_model.py:504`).

✅ **Real curriculum run LAUNCHED** (2026-08-26): `--curriculum --dedup
--no-compile`, batch 4 × accum 8, seq 2048, `--checkpoint-dir
/home/kenpeter/work/x-small/checkpoints/san`, `--save-every 2000`,
resume-from-latest. Reached **step ~50260, loss ~2.9** (~11.6k tok/s, stable
eager). `san_latest.pt` backed up to `san_latest.pt.bak-premix-20260826-123552.pt`
before relaunch. ⚠️ **Run is currently DOWN** — killed during a torch.compile
test (see Pending #2); needs relaunch.

✅ **Compile-mask fix (`san_model.py`)**: attention mask comparison changed from
`torch.equal` (data-dependent graph-break) to identity check vs cached causal
mask — see Training section. Note: torch.compile *still hangs* on autotune for
this model, so **`--no-compile` (eager) is the operational mode**.

🔜 Pending:
1. **Engram fusion** (deferred) — Sinkhorn already Triton-fused (`san_model.py:504`); Engram (`san_model.py:174`) only fires at layers 2,15, so a custom fused gather+matmul kernel is low-ROI. Implement only if more headroom is needed.
2. **Eval** trained SAN (18L) on code/math prompts.
3. **Lift 100W power cap** for more tok/s (heat trade) — optional; currently capped via `nvidia-smi -pl 100` (resets on reboot).
4. **Hermes agent-1 model config** — default switched to `hy3-free` (was `mimo-v2.5-free`); active after gateway restart.
