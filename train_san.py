"""Train the Simple Attention Network (needle port) on x-small's shard data.

SmolLM2 tokenizer (vocab 49152) so it trains directly on the existing .bin
shards (no re-tokenization). Model ~44M params (d_model 384, 27 layers) —
matched to the real needle2 scale (~45M). Faithful needle training: next-token
CE on the main head + optional MTP aux loss.

Checkpointing follows the user rule: NEVER save best.pt — always latest (resume
always from latest). Dataset: flat farm (_shards_final) by default, or the
stratifed domain-tiered curriculum (ported from small) via --curriculum.
"""
import argparse, json, math, os, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from san_model import SimpleAttentionNetwork, SANConfig, make_causal_mask

# ─── Data ────────────────────────────────────────────────────────────
class BinShardDataset(IterableDataset):
    """Sequential .bin shard consumer (x-small style flat farm)."""
    def __init__(self, data_dir: Path, seq_len: int, val_frac: float = 0.0, is_val: bool = False):
        import hashlib
        self.seq_len = seq_len
        shards = sorted(data_dir.glob("*.bin"))
        if not shards:
            raise RuntimeError(f"No .bin shards in {data_dir}")
        sorted_shards = sorted(shards, key=lambda p: hashlib.md5(str(p).encode()).hexdigest())
        n_val = max(1, int(len(sorted_shards) * val_frac))
        self.shards = sorted_shards[-n_val:] if is_val else sorted_shards[:-n_val]
        self.is_val = is_val

    def __iter__(self):
        for si, sp in enumerate(self.shards):
            tokens = np.memmap(str(sp), dtype=np.uint16, mode="r")
            n = len(tokens) - self.seq_len
            if n <= 0:
                continue
            offset = (hash(str(sp)) % max(1, n)) if n > 0 else 0
            starts = list(range(offset, n, self.seq_len))
            if si % 2 == 1:  # paired reversal
                starts.reverse()
            for i in starts:
                chunk = tokens[i:i + self.seq_len + 1]
                if len(chunk) < self.seq_len + 1:
                    continue
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield x, y
            del tokens


def build_dataloader(cfg, is_val=False):
    ds = BinShardDataset(Path(cfg.data_dir), cfg.seq_len, cfg.val_frac, is_val)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=0, pin_memory=True), ds


# ─── Loss ────────────────────────────────────────────────────────────
def chunked_ce(logits, targets, chunk=256):
    """fp32 chunked cross-entropy — avoids materializing [B,S,V] fp32 logits.
    logits (B,S,V), targets (B,S) → mean over non-(-100) tokens."""
    B, S, V = logits.shape
    total, count = 0.0, 0
    for t in range(0, S, chunk):
        ce = F.cross_entropy(
            logits[:, t:t + chunk].float().reshape(-1, V),
            targets[:, t:t + chunk].reshape(-1),
            reduction="sum", ignore_index=-100)
        total += ce
        count += (targets[:, t:t + chunk] != -100).sum().item()
    return total / max(count, 1)


def san_loss(model, x, y, mtp_weight=0.0):
    """Compute next-token CE with faithful needle MTP aux loss (optional)."""
    if mtp_weight > 0:
        main, mtp = model(x, return_mtp=True)
        # main[t] predicts token[t+1] = y[t] (full alignment, no slice)
        loss = chunked_ce(main, y)
        # MTP: at position t, sees x[t] + emb(tok[t+1]) -> predicts tok[t+2].
        # mtp[:, :-1] (B,S-1,V) aligns to target y[:, 1:] (B,S-1).
        loss = loss + mtp_weight * chunked_ce(mtp[:, :-1], y[:, 1:])
        return loss
    logits = model(x)
    return chunked_ce(logits, y)


# ─── LR ──────────────────────────────────────────────────────────────
def get_lr(step, warmup, total, peak, min_lr):
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (peak - min_lr)


@dataclass
class TrainConfig:
    vocab_size: int = 49152
    d_model: int = 384
    num_heads: int = 8
    num_kv_heads: int = 4
    num_layers: int = 27
    seq_len: int = 2048
    val_frac: float = 0.01
    batch_size: int = 4
    grad_accum: int = 8
    max_steps: int = 50000
    lr: float = 4e-4
    min_lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    save_every: int = 2000
    log_every: int = 10
    mtp_weight: float = 0.1
    use_checkpoint: bool = True  # gradient checkpoint MHC layers (cuts long-seq VRAM)
    data_dir: str = "/home/kenpeter/work/data/_shards_final"
    checkpoint_dir: str = "/home/kenpeter/work/checkpoints/xsmall_san"
    dtype: str = "bfloat16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--mtp-weight", type=float, default=0.1)
    ap.add_argument("--data-dir", default="/home/kenpeter/work/data/_shards_final")
    ap.add_argument("--checkpoint-dir", default="/home/kenpeter/work/checkpoints/xsmall_san")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume")
    args = ap.parse_args()

    cfg = TrainConfig(
        max_steps=args.steps, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, seq_len=args.seq_len, mtp_weight=args.mtp_weight,
        data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M",
                                              trust_remote_code=True)
    print(f"Tokenizer vocab: {len(tokenizer)}")

    san_cfg = SANConfig(vocab_size=cfg.vocab_size, d_model=cfg.d_model,
                        num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads,
                        num_layers=cfg.num_layers, max_seq_len=cfg.seq_len,
                        use_checkpoint=cfg.use_checkpoint, dtype=cfg.dtype)
    model = SimpleAttentionNetwork(san_cfg).to(device)
    model.train()
    print(f"PARAMS: {model.count_parameters()/1e6:.2f}M")

    # No decoupled weight decay for simplicity; standard AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(cfg.checkpoint_dir, "san_latest.pt")
    start_step = 0
    ckpt_path = args.resume or (latest_path if os.path.exists(latest_path) else None)
    if ckpt_path and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        try:
            optimizer.load_state_dict(st["optimizer_state_dict"])
        except Exception as e:
            print(f"  optimizer state load failed ({e}) — fresh optimizer")
        start_step = st.get("step", 0)
        print(f"🔁 Resumed from {ckpt_path} (step {start_step})")

    train_loader, ds = build_dataloader(cfg, is_val=False)
    train_iter = iter(train_loader)
    print(f"Shards: {len(ds.shards)}  steps={cfg.max_steps}  eff_batch={cfg.batch_size*cfg.grad_accum}")

    t0 = time.time()
    running = 0.0
    step = start_step
    while step < cfg.max_steps:
        step += 1
        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        acc_loss = 0.0
        for _ in range(cfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            loss = san_loss(model, x, y, cfg.mtp_weight) / cfg.grad_accum
            acc_loss += loss.item() * cfg.grad_accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += acc_loss

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = (cfg.batch_size * cfg.seq_len * cfg.grad_accum * cfg.log_every) / dt
            print(f"step {step:6d} | loss {running/cfg.log_every:.4f} | lr {lr:.2e} | {dt:.1f}s | {tok_s:,.0f} tok/s")
            running = 0.0

        if step % cfg.save_every == 0 or step == cfg.max_steps:
            state = {"step": step, "loss": acc_loss,
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "config": cfg.__dict__}
            tmp = latest_path + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, latest_path)
            print(f"  💾 saved latest (step {step}) -> {latest_path}")

    print("✅ done")


if __name__ == "__main__":
    main()
