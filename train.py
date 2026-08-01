"""
SmolLM2-135M pretraining script.
Transformer++: RMSNorm, SwiGLU, RoPE, GQA.
Saves only latest.pt and best.pt. Resumable.
"""
import os, sys, json, time, math, glob, gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# TF32 for faster matmul on Ampere+ (negligible accuracy loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

# ─── Config ───────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # SmolLM2-135M — fits RTX 4070 Ti 12GB VRAM, no CPU offload needed
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

    # Training (batch 4 × grad-accum 8 = effective 32; maxed for 12GB VRAM —
    # measured: fwd+bwd peak 10.65GB with fused CE; batch 6+ OOMs)
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 100_000
    learning_rate: float = 4e-4
    min_lr: float = 1e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    save_every_n_steps: int = 2000
    log_every_n_steps: int = 10
    val_every_n_steps: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_gradient_checkpointing: bool = True
    compile: bool = False
    seq_len: int = 2048
    val_frac: float = 0.01

    # Checkpointing (stage-isolated: pretrained → sft → dpo)
    checkpoint_dir: Path = Path("/home/kenpeter/work/checkpoints")

    # Data
    data_dir: Path = Path("/home/kenpeter/work/data/_shards_final")

# ─── Architecture (imported from model.py) ────────────────────────────────
from model import SmolLM2

# ─── Data ─────────────────────────────────────────────────────────
class BinShardDataset(IterableDataset):
    def __init__(self, data_dir: Path, seq_len: int, val_frac: float = 0.0, is_val: bool = False):
        self.seq_len = seq_len
        self.shards = sorted(data_dir.glob("*.bin"))
        if not self.shards:
            raise RuntimeError(f"No .bin shards found in {data_dir}")
        self.val_frac = val_frac
        self.is_val = is_val

        # Deterministic stratified split: shuffle by stable hash, then take tail for val
        import hashlib
        sorted_shards = sorted(self.shards, key=lambda p: hashlib.md5(str(p).encode()).hexdigest())
        n_val = max(1, int(len(sorted_shards) * val_frac))
        if is_val:
            self.shards = sorted_shards[-n_val:]
        else:
            self.shards = sorted_shards[:-n_val] if n_val > 0 else sorted_shards

        self.token_dtype = np.uint16
        self.tokens_per_shard = self.shards[0].stat().st_size // np.dtype(self.token_dtype).itemsize

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shards = self.shards
        else:
            # Shard per worker
            per_worker = len(self.shards) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.shards)
            shards = self.shards[start:end]

        for shard_path in shards:
            tokens = np.memmap(str(shard_path), dtype=self.token_dtype, mode="r")
            n = len(tokens) - self.seq_len
            # Random-ish offset per shard to avoid all workers starting at 0
            offset = (hash(str(shard_path)) % max(1, n)) if n > 0 else 0
            for i in range(offset, n, self.seq_len):
                chunk = tokens[i : i + self.seq_len + 1]
                if len(chunk) < self.seq_len + 1:
                    continue
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield x, y
            del tokens
            gc.collect()

def get_dataloader(cfg: TrainConfig, is_val: bool = False):
    ds = BinShardDataset(cfg.data_dir, cfg.seq_len, cfg.val_frac, is_val)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ─── Checkpointing ──────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.checkpoint_dir = cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.checkpoint_dir / "pretrained_latest.pt"
        self.best_path = self.checkpoint_dir / "pretrained_best.pt"
        self.best_loss = float("inf")

    def save(self, model, optimizer, scheduler, step: int, loss: float, is_best: bool = False):
        state = {
            "step": step,
            "loss": loss,
            "best_loss": self.best_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "config": self.cfg.__dict__,
        }
        tmp_path = self.latest_path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, self.latest_path)
        print(f"  💾 Saved latest checkpoint (step {step}, loss {loss:.4f})")

        if is_best or loss < self.best_loss:
            self.best_loss = loss
            os.replace(self.latest_path, self.best_path)
            print(f"  ⭐ Saved best checkpoint (loss {loss:.4f})")

    def load(self, model, optimizer, scheduler):
        ckpt_path = self.latest_path if self.latest_path.exists() else self.best_path
        if not ckpt_path.exists():
            print("  🔄 No checkpoint found, starting from scratch")
            return 0, float("inf")

        print(f"  🔄 Resuming from {ckpt_path.name}")
        state = torch.load(ckpt_path, map_location=self.cfg.device, weights_only=False)
        # Strip _orig_mod prefix from compiled model checkpoints
        model_state = state["model_state_dict"]
        unwanted_prefix = "_orig_mod."
        for k,v in list(model_state.items()):
            if k.startswith(unwanted_prefix):
                model_state[k[len(unwanted_prefix):]] = model_state.pop(k)
        model.load_state_dict(model_state)
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:
            print(f"     ⚠ Optimizer state load failed (switching optimizers?): {e}")
            print(f"     ⚠ Starting optimizer from scratch. Model weights preserved.")
        if scheduler and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if "rng_state" in state:
            try:
                rng_state = state["rng_state"]
                if isinstance(rng_state, list):
                    rng_state = torch.tensor(rng_state, dtype=torch.uint8)
                torch.set_rng_state(rng_state.cpu())
            except Exception as e:
                print(f"     ⚠ RNG restore skipped: {e}")
        if "cuda_rng_state" in state and torch.cuda.is_available():
            try:
                cuda_states = state["cuda_rng_state"]
                if isinstance(cuda_states, list) and cuda_states and isinstance(cuda_states[0], torch.Tensor):
                    torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
                elif isinstance(cuda_states, torch.Tensor):
                    torch.cuda.set_rng_state(cuda_states.cpu())
                else:
                    print(f"     ⚠ CUDA RNG restore skipped: unexpected type {type(cuda_states)}")
            except Exception as e:
                print(f"     ⚠ CUDA RNG restore skipped: {e}")
        step = state.get("step", 0)
        best_loss = state.get("best_loss", float("inf"))
        print(f"     step {step}, best_loss {best_loss:.4f}")
        return step, best_loss

# ─── LR Scheduler ────────────────────────────────────────────────────────
def get_lr(step: int, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    # Cosine decay
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

# ─── Training ───────────────────────────────────────────────────────
@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, max_new: int = 20, device: str = "cuda"):
    """Quick greedy generation for real-situation eval."""
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits, _ = model(input_ids)
        next_tok = logits[0, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok.unsqueeze(0)], dim=1)
    out = tokenizer.decode(input_ids[0].tolist())
    model.train()
    return out

REAL_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The sky is",
    "Water boils at",
    "Once upon a time",
]

@torch.no_grad()
def estimate_loss(model, val_loader, cfg: TrainConfig, max_batches: int = 50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        with torch.autocast(device_type=cfg.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

def train(cfg: TrainConfig):
    print(f"Device: {cfg.device}")
    print(f"Data dir: {cfg.data_dir}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")

    # Verify tokenizer exists
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Model
    model = SmolLM2(cfg)
    model = model.to(cfg.device)
    print(f"Model params: {model.count_parameters():,}")

    # Optimizer — standard AdamW (135M fits easily in VRAM, no 8-bit needed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    scheduler = None  # manual LR

    # Checkpoint manager
    ckpt = CheckpointManager(cfg)
    start_step, best_loss = ckpt.load(model, optimizer, scheduler)

    # Compile AFTER loading checkpoint (avoids _orig_mod key mismatch)
    if cfg.compile and hasattr(torch, "compile") and cfg.device == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Data
    train_loader = get_dataloader(cfg, is_val=False)
    val_loader = get_dataloader(cfg, is_val=True)
    train_iter = iter(train_loader)

    print(f"Train shards: {len(train_loader.dataset.shards)}")
    print(f"Val shards:   {len(val_loader.dataset.shards)}")

    # Scaler for mixed precision (only if float16; bf16 doesn't need scaling)
    use_amp = cfg.device == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler() if use_amp and hasattr(torch.cuda.amp, "GradScaler") and getattr(cfg, 'dtype', 'bfloat16') == 'float16' else None

    # Training loop
    t0 = time.time()
    running_loss = 0.0
    step = start_step
    val_interval = cfg.val_every_n_steps
    stall_count = 0

    while step < cfg.max_steps:
        step += 1
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation micro-batches
        accumulated_loss = 0.0
        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
            ctx = torch.autocast(device_type=cfg.device, dtype=torch.bfloat16 if use_amp else torch.float32)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            accumulated_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # No sleep between microsteps - maximize GPU utilization
            if micro_step < cfg.gradient_accumulation_steps - 1:
                pass

        # Clip + step
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

        # Logging
        if step % cfg.log_every_n_steps == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = (cfg.batch_size * cfg.seq_len * cfg.gradient_accumulation_steps * cfg.log_every_n_steps) / dt
            avg_loss = running_loss / cfg.log_every_n_steps
            print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {dt:.1f}s | {tokens_per_sec:,.0f} tok/s")
            running_loss = 0.0

        # Validation (adaptive: 30 min normally, 15 min if plateau)
        if step % val_interval == 0:
            val_loss = estimate_loss(model, val_loader, cfg)
            print(f"  📊 Val loss: {val_loss:.4f} | best: {best_loss:.4f} | eval_every={val_interval}")
            # Real-situation generation sample
            test_prompt = REAL_PROMPTS[step % len(REAL_PROMPTS)]
            gen = generate_sample(model, tokenizer, test_prompt, max_new=15, device=cfg.device)
            print(f"  💬 Gen: {gen[:80]}")
            if val_loss < best_loss:
                best_loss = val_loss
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= 3 and val_interval > 50:
                    val_interval = max(50, val_interval // 2)
                    print(f"  ⚠ Plateau detected → eval every {val_interval} steps (~15 min)")

        # Checkpointing
        if step % cfg.save_every_n_steps == 0:
            ckpt.save(model, optimizer, scheduler, step, val_loss, is_best=(val_loss == best_loss))

    # Final save
    val_loss = estimate_loss(model, val_loader, cfg)
    ckpt.save(model, optimizer, scheduler, step, val_loss, is_best=(val_loss < best_loss))
    print(f"\n✅ Training complete. Final val loss: {val_loss:.4f}")

if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
