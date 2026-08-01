#!/bin/bash
# Download Phase 2 SFT datasets → /home/kenpeter/work/data/_sft_staging
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /home/kenpeter/work/x-small
source venv_xsmall/bin/activate

exec python3 -u download_sft.py 2>&1 | tee /home/kenpeter/work/sft_download.log
