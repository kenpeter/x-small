"""
SFT (Supervised Fine-Tuning) for SmolLM2-135M.
Loads pretrained checkpoint, fine-tunes on instruction-response pairs.
Only assistant tokens contribute to loss (system/user masked with -100).
Uses pre-tokenized .pt shards from _sft_shards/.
"""
import os, sys, json, time, math, glob, gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

from model import SmolLM2

# ─── Config ───────────────────────────────────────────────────────────────
@dataclass
class SFTConfig:
    vocab_size: int = 49152
    dim: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    intermediate_size: int = 1536
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    base_checkpoint: Path = Path("/home/kenpeter/work/checkpoints/pretrained_best.pt")
    sft_shards_dir: Path = Path("/home/kenpeter/work/data/_sft_shards")
    output_dir: Path = Path("/home/kenpeter/work/checkpoints")
    seq_len: int = 1024
    batch_size: int = 3  # Increased from 2 to improve throughput, still within memory limits
    gradient_accumulation_steps: int = 4
    max_steps: int = 40_000
    learning_rate: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    save_every_n_steps: int = 50
    log_every_n_steps: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Dataset: streaming pre-tokenized .pt shards ─────────────────────────────────────────
class StreamingSFTDataset(torch.utils.data.IterableDataset):
    def __init__(self, shards_dir: Path, seq_len: int = 2048):
        self.shard_files = sorted(shards_dir.glob("shard_*.pt"))
        if not self.shard_files:
            raise RuntimeError(f"No .pt shards found in {shards_dir}")
        print(f"Found {len(self.shard_files)} SFT shards")
        self.seq_len = seq_len

    def _pad_sample(self, s):
        input_ids = torch.tensor(s["input_ids"], dtype=torch.long)
        labels = torch.tensor(s["labels"], dtype=torch.long)

        # FIX: shift labels so model predicts NEXT token, not CURRENT token.
        # Stored labels have labels[j] = input_ids[j] for assistant positions j.
        # We need: position i predicts token at position i+1, so labels[i] = input_ids[i+1]
        # only if position i+1 is part of the assistant response.
        shifted_labels = torch.full_like(labels, -100)
        assistant_positions = (labels != -100).nonzero(as_tuple=True)[0]
        for j in assistant_positions:
            j = j.item()
            if j > 0:
                shifted_labels[j - 1] = input_ids[j]

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            shifted_labels = shifted_labels[:self.seq_len]
        elif len(input_ids) < self.seq_len:
            pad = self.seq_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad, dtype=torch.long)])
            shifted_labels = torch.cat([shifted_labels, torch.full((pad,), -100, dtype=torch.long)])
        return input_ids, shifted_labels

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shard_files = self.shard_files.copy()
        else:
            # Shard across workers
            per_worker = len(self.shard_files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.shard_files)
            shard_files = self.shard_files[start:end]
        import random
        random.shuffle(shard_files)
        for shard_path in shard_files:
            shard = torch.load(shard_path, weights_only=False)
            random.shuffle(shard)
            for s in shard:
                yield self._pad_sample(s)
            del shard
            gc.collect()

# ─── LR Scheduler ─────────────────────────────────────────────────────────
def get_lr(step: int, cfg: SFTConfig):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

# ─── Training ──────────────────────────────────────────────────────────
def train_sft(cfg: SFTConfig):
    print(f"Device: {cfg.device}")
    print(f"Base checkpoint: {cfg.base_checkpoint}")
    print(f"SFT shards: {cfg.sft_shards_dir}")

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocab: {len(tokenizer)}")

    model = SmolLM2(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained or resume SFT
    latest_sft = cfg.output_dir / "sft_latest.pt"
    resume_step = 0
    best_loss = float("inf")
    if latest_sft.exists():
        state = torch.load(latest_sft, map_location=cfg.device, weights_only=False)
        model_state = state["model_state_dict"]
        unwanted_prefix = "_orig_mod."
        for k,v in list(model_state.items()):
            if k.startswith(unwanted_prefix):
                model_state[k[len(unwanted_prefix):]] = model_state.pop(k)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        resume_step = state.get("step", 0)
        best_loss = state.get("best_loss", float("inf"))
        if "rng_state" in state:
            torch.set_rng_state(state["rng_state"].cpu())
        if "cuda_rng_state" in state and cfg.device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() if s.is_cuda else s for s in state["cuda_rng_state"]])
        print(f"  🔄 Resumed SFT from step {resume_step} (best_loss={best_loss:.4f})")
    elif cfg.base_checkpoint.exists():
        state = torch.load(cfg.base_checkpoint, map_location=cfg.device, weights_only=False)
        model_state = state["model_state_dict"]
        # Strip _orig_mod prefix from compiled model checkpoints
        unwanted_prefix = "_orig_mod."
        for k,v in list(model_state.items()):
            if k.startswith(unwanted_prefix):
                model_state[k[len(unwanted_prefix):]] = model_state.pop(k)
        model.load_state_dict(model_state)
        print(f"  🔄 Loaded pretrained from step {state.get('step', 'unknown')}")
    else:
        print(f"  ⚠ No checkpoint found. Training from scratch.")


    # Data
    dataset = StreamingSFTDataset(cfg.sft_shards_dir, seq_len=cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    data_iter = iter(dataloader)
    print(f"Streaming {len(dataset.shard_files)} shards, batch_size={cfg.batch_size}")

    use_amp = cfg.device == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler() if use_amp and hasattr(torch.cuda.amp, "GradScaler") and getattr(cfg, 'dtype', 'bfloat16') == 'float16' else None

    t0 = time.time()
    running_loss = 0.0
    step = resume_step
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    while step < cfg.max_steps:
        step += 1
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accumulated_loss = 0.0
        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(cfg.device), y.to(cfg.device)
            # Skip samples with no valid response tokens (all labels = -100)
            if (y != -100).sum() == 0:
                continue
            ctx = torch.autocast(device_type=cfg.device, dtype=torch.bfloat16 if use_amp else torch.float32)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            if torch.isnan(loss):
                continue
            accumulated_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += accumulated_loss

        if step % cfg.log_every_n_steps == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = (cfg.batch_size * cfg.seq_len * cfg.gradient_accumulation_steps * cfg.log_every_n_steps) / dt
            avg_loss = running_loss / cfg.log_every_n_steps
            print(f"step {step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.1f}s | {tokens_per_sec:,.0f} tok/s")
            running_loss = 0.0

        if step % cfg.save_every_n_steps == 0:
            val_loss = avg_loss
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            state = {
                "step": step,
                "loss": val_loss,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else None,
            }
            tmp = cfg.output_dir / "sft_latest.tmp"
            torch.save(state, tmp)
            os.replace(tmp, cfg.output_dir / "sft_latest.pt")
            if is_best:
                os.replace(cfg.output_dir / "sft_latest.pt", cfg.output_dir / "sft_best.pt")
                print(f"  ⭐ Saved best SFT (loss {val_loss:.4f})")
            else:
                print(f"  💾 Saved latest SFT (loss {val_loss:.4f})")

    state = {
        "step": step,
        "loss": running_loss,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.__dict__,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else None,
    }
    torch.save(state, cfg.output_dir / "sft_final.pt")
    print(f"\n✅ SFT complete. sft_final.pt | best={best_loss:.4f}")

if __name__ == "__main__":
    cfg = SFTConfig()
    train_sft(cfg)
