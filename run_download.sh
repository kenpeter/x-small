#!/bin/bash
# Wrapper to start downloads with correct environment
unset ENABLE_CUDA_GRAPH
export HF_XET_HIGH_PERFORMANCE=1
cd /home/kenpeter/work/small
source venv/bin/activate
exec python3 download_3workers_direct.py > /home/kenpeter/work/small/download.log 2>&1
