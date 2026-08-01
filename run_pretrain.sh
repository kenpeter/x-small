#!/bin/bash
unset ENABLE_CUDA_GRAPH
unset ENABLE_HYDRA_PIKIA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/kenpeter/work/small/venv/bin/activate
cd /home/kenpeter/work/small

echo "GPU status before start:"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit --format=csv,noheader
echo ""
echo "Tip: If you want a cooler run, cap power with: sudo nvidia-smi -pl 200"
echo "Starting pretrain at $(date)"
echo ""

exec python3 pretrain_megatrain.py \
  --batch-size 8 \
  --grad-accum 4 \
  --num-steps 60000 \
  --log-interval 400 \
  --save-interval 3000 \
  --warmup-steps 1000 \
  --lr 3e-4 \
  --dtype bfloat16 \
  --output-dir /home/kenpeter/work/checkpoints \
  --resume /home/kenpeter/work/checkpoints/megatrain_latest.pt \
  > /home/kenpeter/work/train.log 2>&1
