#!/bin/bash
# Pretrain SmolLM2-135M — pure GPU, no MegaTrain/CPU offload.
# Effective batch 32 (2 × 16 grad-accum), fits RTX 4070 Ti 12GB.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kenpeter/work/x-small
source venv_xsmall/bin/activate

exec python3 -u train.py 2>&1 | tee /home/kenpeter/work/train.log
