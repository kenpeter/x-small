#!/usr/bin/env python3
"""Test: 3 tokens in parallel via curl — with stderr visible."""
import os, time, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor

TOKENS = [
    os.environ.get("HF_TOKEN_1", "hf_XXXX"),
    os.environ.get("HF_TOKEN_2", "hf_XXXX"),
    os.environ.get("HF_TOKEN_3", "hf_XXXX"),
]
REPO = "princeton-nlp/QuRatedPajama-260B"
STAGING = "/tmp/token_test"
shutil.rmtree(STAGING, ignore_errors=True)
os.makedirs(STAGING, exist_ok=True)

def dl(idx):
    tok = TOKENS[idx]
    fname = f"train-{100+idx:05d}-of-04022.parquet"
    local = os.path.join(STAGING, fname)
    t0 = time.time()
    cmd = ["curl", "-sS", "-o", local, "--max-time", "75",
           "-H", f"Authorization: Bearer {tok}",
           "-L", f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{fname}",
           "-r", "0-52428799"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    got = os.path.getsize(local) if os.path.exists(local) else 0
    err = r.stderr.strip()[:120] if r.stderr.strip() else "ok"
    return idx, got / 1e6, time.time() - t0, err

t0 = time.time()
with ThreadPoolExecutor(max_workers=3) as ex:
    for idx, mb, dt, err in ex.map(dl, [0, 1, 2]):
        print(f"token{idx}: {mb:.0f}MB in {dt:.0f}s ({mb/dt:.1f} MB/s) | {err}", flush=True)
el = time.time() - t0
total = sum(os.path.getsize(os.path.join(STAGING, f)) for f in os.listdir(STAGING)) / 1e6
print(f"TOTAL: {total:.0f}MB in {el:.0f}s = {total/el:.2f} MB/s aggregate")
