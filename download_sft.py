#!/usr/bin/env python3
"""
Download Phase 2 SFT datasets with 3 parallel workers.
Datasets: OpenHermes-2.5, OpenOrca, Ultrachat, Alpaca-GPT4, Code-Alpaca
"""
import os, sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SFT_DATASETS = {
    "openhermes":    {"repo": "teknium/OpenHermes-2.5",      "split": "train", "streaming": False},
    "openorca":      {"repo": "Open-Orca/OpenOrca",          "split": "train", "streaming": False},
    "ultrachat":     {"repo": "stingning/ultrachat",         "split": "train", "streaming": True},
    "alpaca_gpt4":   {"repo": "vicgalle/alpaca-gpt4",        "split": "train", "streaming": False},
    "code_alpaca":   {"repo": "sahil2801/CodeAlpaca-20k",    "split": "train", "streaming": False},
}

STAGING_DIR = Path("/home/kenpeter/work/data/_sft_staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

def download_one(name: str, info: dict):
    t0 = time.time()
    out_dir = STAGING_DIR / name
    out_dir.mkdir(exist_ok=True)
    print(f"[{name}] Starting download from {info['repo']} ...")
    try:
        ds = load_dataset(
            info["repo"],
            split=info["split"],
            streaming=info["streaming"],
        )
        # Save as JSONL shards
        shard_size = 10000
        shard_idx = 0
        shard_lines = []
        total = 0
        for i, ex in enumerate(ds):
            shard_lines.append(json.dumps(ex, ensure_ascii=False))
            total += 1
            if len(shard_lines) >= shard_size:
                out_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(shard_lines) + "\n")
                shard_idx += 1
                shard_lines = []
                if shard_idx % 10 == 0:
                    print(f"[{name}] Saved {shard_idx} shards ({total:,} samples)...")
        if shard_lines:
            out_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(shard_lines) + "\n")
        elapsed = time.time() - t0
        print(f"[{name}] ✅ DONE: {total:,} samples in {shard_idx+1} shards | {elapsed:.0f}s")
        return {"name": name, "status": "ok", "samples": total, "shards": shard_idx+1, "time": elapsed}
    except Exception as e:
        print(f"[{name}] ❌ FAILED: {e}")
        return {"name": name, "status": "error", "error": str(e)}

if __name__ == "__main__":
    print(f"SFT staging: {STAGING_DIR}")
    print(f"Workers: 3 | Datasets: {list(SFT_DATASETS.keys())}\n")
    results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_one, name, info): name for name, info in SFT_DATASETS.items()}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
    print("\n" + "="*60)
    print("SFT DOWNLOAD SUMMARY")
    print("="*60)
    for r in results:
        if r["status"] == "ok":
            print(f"  ✅ {r['name']:15s} | {r['samples']:>10,} samples | {r['shards']:>4} shards | {r['time']:.0f}s")
        else:
            print(f"  ❌ {r['name']:15s} | ERROR: {r.get('error','')}")
