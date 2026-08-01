"""
Tokenize Phase-2 SFT datasets into binary shards.
Format: chat template with <|im_start|>user/assistant/system<|im_end|>
Only assistant tokens have non-masked labels (-100 for user/system).
"""
import os, sys, json, time, gc
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────
SFT_STAGING = Path("/home/kenpeter/work/data/_sft_staging")
OUT_DIR = Path("/home/kenpeter/work/data/_sft_shards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARD_SIZE = 100_000  # samples per shard
MAX_SEQ_LEN = 2048

# Chat template for SmolLM2
def format_chat(messages: list) -> str:
    """messages: list of {"role": "system|user|assistant", "content": str}"""
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return text

def format_openhermes(ex: dict) -> list:
    msgs = ex.get("messages", [])
    if not msgs:
        return None
    # Ensure system message
    if msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
    return msgs

def format_openorca(ex: dict) -> list:
    system = ex.get("system_prompt", "You are a helpful assistant.")
    question = ex.get("question", ex.get("prompt", ""))
    response = ex.get("response", ex.get("answer", ""))
    if not question or not response:
        return None
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]

def format_alpaca(ex: dict) -> list:
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")
    output = ex.get("output", "")
    if not instruction or not output:
        return None
    if inp:
        instruction += f"\n{inp}"
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

def format_ultrachat(ex: dict) -> list:
    data = ex.get("data", [])
    if not data or len(data) < 2:
        return None
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i, turn in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": turn})
    return msgs

DATASET_FORMATTERS = {
    "openhermes":    format_openhermes,
    "openorca":      format_openorca,
    "ultrachat":     format_ultrachat,
    "alpaca_gpt4":   format_alpaca,
    "code_alpaca":   format_alpaca,
}

def tokenize_sample(tokenizer, messages: list, max_len: int = MAX_SEQ_LEN):
    """Tokenize chat messages. Return input_ids and labels (assistant only)."""
    # Build text with role markers to find assistant boundaries
    full_text = ""
    role_boundaries = []  # (start_char, end_char, role)
    for msg in messages:
        start = len(full_text)
        full_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        end = len(full_text)
        role_boundaries.append((start, end, msg["role"]))

    # Tokenize full text
    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding.get("offset_mapping", None)

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        if offsets:
            offsets = offsets[:max_len]

    # Build labels: -100 for non-assistant, actual token_id for assistant
    labels = [-100] * len(input_ids)
    if offsets:
        for i, (char_start, char_end) in enumerate(offsets):
            # Find which role this token belongs to
            for rstart, rend, role in role_boundaries:
                if char_start >= rstart and char_end <= rend and role == "assistant":
                    labels[i] = input_ids[i]
                    break
    else:
        # Fallback: find assistant content text and mark tokens
        for rstart, rend, role in role_boundaries:
            if role == "assistant":
                # Rough: mark all tokens after finding "<|im_start|>assistant"
                pass  # skip fallback for now

    return input_ids, labels

def main():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    print(f"Tokenizer vocab: {len(tokenizer)}")

    all_samples = []
    shard_idx = 0
    total_samples = 0
    total_tokens = 0

    for dataset_name, formatter in DATASET_FORMATTERS.items():
        ds_dir = SFT_STAGING / dataset_name
        if not ds_dir.exists():
            print(f"  ⚠ {dataset_name}: dir not found, skipping")
            continue

        jsonl_files = sorted(ds_dir.glob("shard_*.jsonl"))
        print(f"\n📥 [{dataset_name}] {len(jsonl_files)} files...")

        for jfile in tqdm(jsonl_files, desc=dataset_name):
            with open(jfile, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    messages = formatter(ex)
                    if messages is None:
                        continue

                    input_ids, labels = tokenize_sample(tokenizer, messages, MAX_SEQ_LEN)
                    if len(input_ids) < 2:
                        continue

                    all_samples.append({
                        "input_ids": input_ids,
                        "labels": labels,
                        "dataset": dataset_name,
                    })
                    total_samples += 1
                    total_tokens += len(input_ids)

                    if len(all_samples) >= SHARD_SIZE:
                        # Save shard
                        out_path = OUT_DIR / f"shard_{shard_idx:05d}.pt"
                        torch.save(all_samples, out_path)
                        print(f"  💾 Saved {out_path.name}: {len(all_samples)} samples")
                        all_samples = []
                        shard_idx += 1
                        gc.collect()

    # Save remaining
    if all_samples:
        out_path = OUT_DIR / f"shard_{shard_idx:05d}.pt"
        torch.save(all_samples, out_path)
        print(f"  💾 Saved {out_path.name}: {len(all_samples)} samples")
        shard_idx += 1

    print(f"\n✅ SFT TOKENIZATION COMPLETE")
    print(f"   Shards: {shard_idx}")
    print(f"   Samples: {total_samples:,}")
    print(f"   Tokens: {total_tokens:,}")
    print(f"   Avg seq len: {total_tokens/max(total_samples,1):.0f}")
    print(f"   Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
