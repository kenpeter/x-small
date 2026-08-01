#!/usr/bin/env python3
"""Tokenize ALL filtered data into binary shards. Cap at 3.3B tokens."""
import os, sys, json, time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER_NAME = "HuggingFaceTB/cosmo2-tokenizer"
FILTERED_DIR = Path("/home/kenpeter/work/data/_filtered_new")
OUTPUT_DIR = Path("/home/kenpeter/work/data/_shards_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENS_PER_SHARD = 256_000_000  # 512MB each (uint16)
token_target = 3_300_000_000

# Priority: best quality first, but include code for diversity
DATASET_PRIORITY = [
    "fineweb-edu",      # 3M docs, diverse educational
    "finemath-3plus",   # 2M docs, math/science
    "stack-python",     # 3M docs, code (FIXED filter!)
    "open-web-math",    # 33K docs, advanced math
    "finemath",         # 17K docs, math problems
    "cosmopedia",       # 5.5K docs, synthetic (lowest)
]

MAX_TEXT_LEN = 50_000
BATCH_SIZE = 2000

def tokenize_dataset(jsonl_files, tokenizer, target_tokens, shard_buf, shard_idx):
    total_tokens = 0
    total_docs = 0
    
    for jsonl_path in jsonl_files:
        print(f"  {jsonl_path.name}...", end="", flush=True)
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            batch_texts = []
            for line in f:
                data = json.loads(line)
                text = data.get('text', '')[:MAX_TEXT_LEN]
                if len(text) < 20:
                    continue
                batch_texts.append(text)
                
                if len(batch_texts) >= BATCH_SIZE:
                    encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
                    for ids in encoded["input_ids"]:
                        arr = np.array(ids, dtype=np.uint16)
                        shard_buf.append(arr)
                        total_tokens += len(arr)
                        total_docs += 1
                    
                    while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                        shard_idx = _flush_shard(shard_buf, shard_idx)
                    
                    batch_texts = []
                    
                    if total_tokens >= target_tokens:
                        print(f" HIT TARGET")
                        return total_tokens, total_docs, shard_buf, shard_idx
        
        # Flush remaining batch
        if batch_texts:
            encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
            for ids in encoded["input_ids"]:
                arr = np.array(ids, dtype=np.uint16)
                shard_buf.append(arr)
                total_tokens += len(arr)
                total_docs += 1
            
            while shard_buf and sum(len(a) for a in shard_buf) >= TOKENS_PER_SHARD:
                shard_idx = _flush_shard(shard_buf, shard_idx)
        
        print(f" {total_tokens:,} tok")
        if total_tokens >= target_tokens:
            return total_tokens, total_docs, shard_buf, shard_idx
    
    return total_tokens, total_docs, shard_buf, shard_idx

def _flush_shard(shard_buf, shard_idx):
    all_tokens = np.concatenate(shard_buf)
    if len(all_tokens) >= TOKENS_PER_SHARD:
        out_path = OUTPUT_DIR / f"shard_{shard_idx:06d}.bin"
        all_tokens[:TOKENS_PER_SHARD].tofile(str(out_path))
        print(f"\n  → Shard {shard_idx}: {TOKENS_PER_SHARD:,} tok ({out_path.stat().st_size/1e6:.1f}MB)")
        remainder = [all_tokens[TOKENS_PER_SHARD:]] if len(all_tokens) > TOKENS_PER_SHARD else []
        shard_buf[:] = remainder
        return shard_idx + 1
    shard_buf[:] = [all_tokens]
    return shard_idx

def main():
    print("=" * 60)
    print("TOKENIZE FILTERED DATA → EXACTLY 3.3B TOKENS (100% FILTERED)")
    print("=" * 60)
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print(f"  Vocab: {len(tokenizer)}")
    
    shard_buf = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0
    
    for ds in DATASET_PRIORITY:
        ds_dir = FILTERED_DIR / ds
        if not ds_dir.exists():
            print(f"\n[SKIP] {ds}: not found")
            continue
        
        jsonl_files = sorted(ds_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue
        
        remaining = token_target - total_tokens
        if remaining <= 0:
            print(f"\n[STOP] Target reached.")
            break
        
        print(f"\n[{ds.upper()}] {len(jsonl_files)} files | Need: {remaining:,} more tokens")
        
        ds_tokens, ds_docs, shard_buf, shard_idx = tokenize_dataset(
            jsonl_files, tokenizer, remaining, shard_buf, shard_idx
        )
        total_tokens += ds_tokens
        total_docs += ds_docs
        
        print(f"  Added: {ds_tokens:,} tok | {ds_docs:,} docs")
        print(f"  Running: {total_tokens:,} / {token_target:,} ({100*total_tokens/token_target:.1f}%)")
    
    # Flush final partial shard
    if shard_buf and len(shard_buf[0]) > 0:
        final = np.concatenate(shard_buf)
        out_path = OUTPUT_DIR / f"shard_{shard_idx:06d}.bin"
        final.tofile(str(out_path))
        print(f"\n  → Final shard {shard_idx}: {len(final):,} tokens")
        shard_idx += 1
        total_tokens += len(final)
    
    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)
    print(f"Shards: {shard_idx}")
    print(f"Tokens: {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    print(f"Docs: {total_docs:,}")
    print(f"Avg: {total_tokens//max(total_docs,1):,} tok/doc")
    
    for s in sorted(OUTPUT_DIR.glob("shard_*.bin")):
        size_mb = s.stat().st_size / 1e6
        tokens = int(size_mb * 1e6 / 2)
        print(f"  {s.name}: {size_mb:.1f}MB (~{tokens:,} tok)")

if __name__ == "__main__":
    main()
