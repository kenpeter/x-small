# SAN x-small — Abstraction

> Abstraction of the project and the agent session that produced this release.

## What SAN is
SAN (Simple Attention Network) is a ~31M-parameter custom language model — a faithful
PyTorch port of the `needle` architecture (arXiv:2607.18363) from JAX/Flax. It is trained
locally on a single **RTX 4070 Ti 12 GB**.

Components (1:1 with the JAX source):
- **ZCRMSNorm** — zero-centered RMSNorm
- **HadamardMLP** — fixed Walsh–Hadamard transform MLP (no dense FFN weights)
- **MultiHeadAttention** — GQA + q/k RMSNorm + RoPE + learned gate
- **Engram** — hashed n-gram KV memory tables with conv taps, injected at sites
- **MHC** — Multi-Lane Hyper-Connections with Sinkhorn routing across 4 residual lanes
- **MTP** — multi-token-prediction head
- Tied embeddings + causal/packing mask

## Trained configuration (this release)
| field | value |
|---|---|
| vocab_size | 49152 (SmolLM2-135M tokenizer) |
| d_model | 384 |
| num_heads / num_kv_heads | 8 / 4 (GQA) |
| num_layers | 12 |
| max_seq_len | 2048 |
| dtype | bfloat16 |
| engram_orders / slots / layers | (2,3) / 8192 / (2,15) → active (2,) |
| mhc_lanes | 4 |
| mtp_weight | 0.1 |

Training: curriculum over **16 domains**, target **100w (1,000,000) steps**,
`--batch-size 8 --grad-accum 8` (eff batch 64). GPU power cap **100 W** (cool) /
**200 W** (sweet spot; ~31.5k tok/s @200W, ~23k @100W).

## Session log (this agent run)
1. **Resume @100W** — `nvidia-smi -pl 100`, resumed from `san_latest.pt` (step 132,706),
   target `--steps 1000000`. Stepping confirmed (~21–23k tok/s).
2. **"Plateau?"** — analyzed eval perplexity (the real signal, not the flat training-loss):
   - step 63,663 → mean ppl **53.69** (code 13.76 / math 36.41 / prose 110.89)
   - step 110,938 → mean ppl **25.38** (code 11.75 / math 30.32 / prose 34.07)
   - step 123,141 → mean ppl **23.74** (code 10.25 / math 29.62 / prose 31.34)
   - Conclusion: **not a plateau** — eval ppl still improving. The flat *training* loss
     (~2.0 since ~105k) is curriculum-mix masking, not model stall. The ~2.0 floor is:
     (a) model-capacity + data-entropy floor for 31M on a diverse 16-domain mix,
     (b) the harder MTP aux head (~2.35), (c) curriculum re-glide sawtooth.
     LR is still near peak (cosine over 1M steps, only 13.5% in) — LR is *not* the cause.
3. **Save + stop + release GPU** — SIGTERM saved checkpoint at **step 137,260**
   (`san_latest.pt`, `✅ done`), released GPU (idle).
4. **Release decision** — HF token provided; decided: **code + this abstraction → GitHub**
   (`kenpeter/x-small`), **model weights + latest `.pt` → HuggingFace** (`kenpeter123/san-xsmall`).

## Key operating insights
- **Eval perplexity is the usable signal** at this scale; generated text loops/decays
  regardless at 31M — do not read sample text as a quality verdict.
- **Training-loss flat ≈ curriculum masking**, not stall — trust eval ppl.
- **Power cap resets to ~285 W on reboot** — reapply after restart.
- **Resume is the normal path**; `--num-layers` is resume-safe (per-layer params sliced).
- **Checkpoint save requires SIGTERM** (saves at next step boundary then exits) — there is
  no save-and-keep-running signal.

## Release layout
- **GitHub `kenpeter/x-small`**: training/inference code (`train_san.py`, `san_model.py`,
  `san_triton.py`, eval scripts), this `abstraction.md`, README, LICENSE.
- **HuggingFace `kenpeter123/san-xsmall`**: `pytorch_model.bin` (model weights),
  `san_latest.pt` (raw checkpoint, step 137,260), `config.json`, `tokenizer/`
  (SmolLM2-135M), `san_model.py` + `load_example.py`, model card.

## Status at release
- Last training step: **137,260** (of 1,000,000 target).
- Last eval (step 123,141): mean ppl **23.74** (code 10.25 / math 29.62 / prose 31.34);
  training loss @ckpt 2.04.
- No eval run between 123,141 and 137,260 — recommend an eval to confirm the trend if
  training resumes.
