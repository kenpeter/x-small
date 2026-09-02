# SAN x-small

Simple Attention Network — a ~31M-param custom LLM, faithful PyTorch port of the `needle`
architecture (arXiv:2607.18363). Trained locally on an RTX 4070 Ti 12 GB.

**Released weights:** 🤗 [kenpeter123/san-xsmall](https://huggingface.co/kenpeter123/san-xsmall)

## Architecture
| | |
|---|---|
| params | ~31M |
| d_model | 384 |
| layers | 12 |
| heads / KV heads | 8 / 4 (GQA) |
| vocab | 49152 (SmolLM2-135M tokenizer) |
| context | 2048 |
| dtype | bfloat16 |

Components: ZCRMSNorm, HadamardMLP (Walsh–Hadamard transform), GQA + RoPE + q/k RMSNorm +
learned gate, Engram n-gram KV memory, Multi-Lane Hyper-Connections (MHC) with Sinkhorn
routing, MTP head, tied embeddings.

## Layout
- `san_model.py` — `SimpleAttentionNetwork` + `SANConfig` (the architecture)
- `san_triton.py` — optional Triton accelerators (guarded; model runs without Triton via a
  pure-PyTorch fallback)
- `train_san.py` — training loop (curriculum, MTP aux loss, resume-safe)
- `eval_san_gen.py` / `eval_san_sample.py` — perplexity eval / qualitative sampling
- `bench_san_tput.py`, `probe_mem*.py`, `tokenize_final.py` — utilities

## Train
```bash
cd /home/kenpeter/work/x-small
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 venv_xsmall/bin/python -u train_san.py \
  --resume checkpoints/san/san_latest.pt --curriculum --num-layers 12 --batch-size 8 --steps 1000000
```
Resume is the normal path; `--num-layers` is resume-safe.

## Eval
```bash
CUDA_VISIBLE_DEVICES=0 venv_xsmall/bin/python -u eval_san_gen.py   # perplexity (the real metric)
```

## Load the released weights
See the HuggingFace repo for `load_example.py`. In short:
```python
from san_model import SimpleAttentionNetwork, SANConfig
import torch
cfg = SANConfig(num_layers=12)        # matches the released checkpoint
model = SimpleAttentionNetwork(cfg)
sd = torch.load("pytorch_model.bin")   # or san_latest.pt -> ["model_state_dict"]
model.load_state_dict(sd)
model.eval()
```

## Notes
- Eval perplexity is the usable quality signal at this scale; sample text loops regardless.
- Training loss flat ≈ curriculum masking, not model stall.
- See `abstraction.md` for the full project/ops writeup.

## License
MIT.
