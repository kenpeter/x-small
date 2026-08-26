#!/bin/bash
# Self-gating SAN re-eval watchdog. Fires once: when training has advanced
# >=2000 steps past the gold-boost resume (step 14000 -> >=16000), runs both
# greedy + sampled evals, prints a compact verdict, then marks done (silent after).
set -e
cd /home/kenpeter/work/x-small
CKPT=checkpoints/san/san_latest.pt
MARKER=evaluations/.reeval_done
STEP=$(./venv_xsmall/bin/python -c "import torch;print(torch.load('$CKPT',map_location='cpu')['step'])" 2>/dev/null || echo 0)
[ "$STEP" -lt 16000 ] && exit 0
[ -f "$MARKER" ] && exit 0
GEN=$(./venv_xsmall/bin/python eval_san_gen.py 2>/dev/null)
SAMP=$(./venv_xsmall/bin/python eval_san_sample.py 2>/dev/null)
PPL=$(echo "$GEN" | grep -i "mean ppl" | tail -1)
echo "SAN re-eval @ step $STEP (gold-boosted run)"
echo "$PPL"
echo "--- sampled excerpts ---"
echo "$SAMP" | grep -A2 "^### \[" | head -48
echo "$SAMP" | grep "SAVED ->" | tail -1
touch "$MARKER"
