"""Train the Simple Attention Network (needle port) on x-small's shard data.

SmolLM2 tokenizer (vocab 49152) so it trains directly on the existing .bin
shards (no re-tokenization). Model ~44M params (d_model 384, 27 layers) —
matched to the real needle2 scale (~45M). Faithful needle training: next-token
CE on the main head + optional MTP aux loss.

Checkpointing follows the user rule: NEVER save best.pt — always latest (resume
always from latest). Dataset: flat farm (_shards_final) by default, or the
stratifed domain-tiered curriculum (ported from small) via --curriculum.
"""
import argparse, json, math, os, random, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
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
        # IterableDataset + num_workers REQUIRES manual sharding, or every worker
        # yields the identical stream (→ each step sees the same batches → "frozen"
        # loss + no dataset consumption). Split shards across worker ranks.
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None:
            shards = shards[worker_info.id::worker_info.num_workers]
        for si, sp in enumerate(shards):
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
    if cfg.curriculum:
        return build_curriculum_dataloader(cfg, is_val)
    ds = BinShardDataset(Path(cfg.data_dir), cfg.seq_len, cfg.val_frac, is_val)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=8, pin_memory=True,
                      persistent_workers=True, prefetch_factor=4), ds


# ============================================================================
# DoReMi-lite G1-G4 stratified curriculum (ported from small/pretrain_megatrain.py)
#
# G1 Boundary Sharpening: easy-heavy start -> hard-heavy end (smooth)
# G2 Cyclic Scheduling: periodic easy-data review wave (anti-forgetting)
# G3 Curriculum Continuity: ratios shift gradually, re-glided every
#    CURRICULUM_UPDATE_INTERVAL steps (no cliff switches)
# G4 Local Diversity: JIT windowed shuffle in _build_stratified_order
# ---------------------------------------------------------------
# HOT-RELOAD (agent-in-the-loop DoReMi): if curriculum_boost.json exists next to
# this module, its per-domain multipliers OVERRIDE WEB_BOOST (missing domains keep
# WEB_BOOST). The file is re-read at every re-glide — edit the JSON and the next
# re-glide applies it. No restart.
# ============================================================================

SHARD_DIRS = {
    "math_easy":     Path("/home/kenpeter/work/data/_shards_math_easy"),
    "math_medium":   Path("/home/kenpeter/work/data/_shards_math_medium"),
    "synth_easy":    Path("/home/kenpeter/work/data/_shards_synth_easy"),
    "synth_medium":  Path("/home/kenpeter/work/data/_shards_synth_medium"),
    "synth_hard":    Path("/home/kenpeter/work/data/_shards_synth_hard"),
    "code_easy":     Path("/home/kenpeter/work/data/_shards_code_easy"),
    "code_medium":   Path("/home/kenpeter/work/data/_shards_code_medium"),
    "code_hard":     Path("/home/kenpeter/work/data/_shards_code_hard"),
    "web_easy":      Path("/home/kenpeter/work/data/_shards_web_easy"),
    "web_medium":    Path("/home/kenpeter/work/data/_shards_web_medium"),
    "web_hard":      Path("/home/kenpeter/work/data/_shards_web_hard"),
    "math_hard":     Path("/home/kenpeter/work/data/_shards_math_hard"),
    "reformat_easy": Path("/home/kenpeter/work/data/_shards_reformat_easy"),
    "gold_hard":     Path("/home/kenpeter/work/data/_shards_gold"),
    "web_gold":      Path("/home/kenpeter/work/data/_shards_web_gold"),
    "code_gold":     Path("/home/kenpeter/work/data/_shards_code_gold"),
}

CURRICULUM_UPDATE_INTERVAL = 2000  # steps between ratio rebuilds

EASY_SPLIT = {"math_easy": 0.35, "web_easy": 0.35, "synth_easy": 0.125, "code_easy": 0.10, "reformat_easy": 0.075}
MED_SPLIT = {"math_medium": 0.25, "web_medium": 0.25, "synth_medium": 0.333, "code_medium": 0.167}
HARD_SPLIT = {"math_hard": 0.40, "web_hard": 0.15, "synth_hard": 0.17, "code_hard": 0.07, "gold_hard": 0.06, "web_gold": 0.03, "code_gold": 0.12}

DOMAIN_TIER = {}
for _d in EASY_SPLIT:   DOMAIN_TIER[_d] = "_easy"
for _d in MED_SPLIT:    DOMAIN_TIER[_d] = "_medium"
for _d in HARD_SPLIT:   DOMAIN_TIER[_d] = "_hard"

WEB_BOOST = 1.5
WEB_BOOST_FILE = Path(__file__).parent / "curriculum_boost.json"


def _smooth_tier_weights(t):
    tt = t if t < 0.5 else 1.0 - t
    w_easy = 0.05 + 0.25 * (1 - tt)
    w_hard = min(0.70, 0.25 + 0.45 * tt)
    w_med = max(0.0, 1.0 - w_easy - w_hard)
    return w_easy, w_med, w_hard


def get_curriculum_ratios(step, total_steps):
    t = min(1.0, max(0.0, step / total_steps))
    w_easy, w_med, w_hard = _smooth_tier_weights(t)
    # G2: cyclic review wave — easy data gets a periodic boost
    cycle = max(1, total_steps // 8)
    review = 0.12 * (0.5 - 0.5 * math.cos(2 * math.pi * step / cycle))
    w_easy = min(0.5, w_easy + review)
    s = w_easy + w_med + w_hard
    w_easy, w_med, w_hard = w_easy / s, w_med / s, w_hard / s

    ratios = {}
    for dom, frac in EASY_SPLIT.items():
        ratios[dom] = round(w_easy * frac, 4)
    for dom, frac in MED_SPLIT.items():
        ratios[dom] = round(w_med * frac, 4)
    for dom, frac in HARD_SPLIT.items():
        ratios[dom] = round(w_hard * frac, 4)

    boost = {}
    try:
        if WEB_BOOST_FILE.exists():
            with open(WEB_BOOST_FILE) as f:
                boost = json.load(f)
    except Exception:
        boost = {}
    tier_totals = {}
    for tier in ("_easy", "_medium", "_hard"):
        tier_totals[tier] = sum(v for k, v in ratios.items() if DOMAIN_TIER.get(k) == tier)
    for dom in ratios:
        dom_boost = WEB_BOOST if dom.startswith("web") else 1.0
        dom_boost = boost.get(dom, dom_boost)
        ratios[dom] = round(ratios[dom] * dom_boost, 4)
    if WEB_BOOST != 1.0 or boost:
        for tier in ("_easy", "_medium", "_hard"):
            doms = [d for d in ratios if DOMAIN_TIER.get(d) == tier]
            s = sum(ratios[d] for d in doms)
            if s <= 0:
                continue
            for d in doms:
                ratios[d] = round(ratios[d] * tier_totals[tier] / s, 4)
            resid = tier_totals[tier] - sum(ratios[d] for d in doms)
            if resid:
                ratios[max(doms, key=ratios.get)] += resid
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        ratios[max(ratios, key=ratios.get)] += 1.0 - total
    return {k: v for k, v in ratios.items() if v > 0}


def _load_shard_list(shards_dir: Path, seq_len: int):
    shard_paths = sorted(shards_dir.glob("*.bin"))
    shard_paths = [p for p in shard_paths if p.stat().st_size > 0]
    entries = []
    total = 0
    for p in shard_paths:
        n_tokens = p.stat().st_size // 2
        # Only index sequences that have a full seq_len+1 continuation window
        # (used by __getitem__ via _fetch_tokens(plus_one=True)). This drops the
        # trailing partial window when a shard ends exactly on a seq_len boundary;
        # otherwise the plus_one read would overrun the file (mmap error).
        n_seqs = (n_tokens - 1) // seq_len
        if n_seqs == 0:
            continue
        entries.append((p, n_seqs, total))
        total += n_seqs
    return entries, total


def _hash_13gram(tokens: np.ndarray) -> int:
    if len(tokens) < 13:
        return 0
    front = tokens[:13].tobytes()
    back = tokens[-13:].tobytes()
    return hash((front, back))


class StratifiedShardDataset(Dataset):
    """Stratified multi-domain loader with ratio-driven interleave (DoReMi-lite).

    Yields dicts {"input_ids": x, "labels": y, "domain": dom} where x (B seq_len)
    and y = x shifted one token (next-token targets), matching x-small semantics.
    Only the strided `seq_len` windows are indexed; dedup scans are optional.
    """

    def __init__(self, shard_dirs: dict, seq_len: int = 2048,
                 ratios: dict = None, dedup: bool = False):
        self.seq_len = seq_len
        self.ratios = ratios or {}
        self.domains = []
        self.domain_entries = {}
        self.domain_totals = {}
        grand_total = 0

        for domain, dpath in shard_dirs.items():
            if not dpath.exists():
                print(f"  ! curriculum: shard dir missing '{domain}': {dpath} — skipping")
                continue
            entries, total = _load_shard_list(dpath, seq_len)
            if total == 0:
                continue
            self.domains.append(domain)
            self.domain_entries[domain] = entries
            self.domain_totals[domain] = total
            grand_total += total

        if not self.domains:
            raise FileNotFoundError("No valid shard directories found for curriculum")

        self.index = []
        self.domain_offsets = {}
        cursor = 0
        for domain in self.domains:
            self.domain_offsets[domain] = cursor
            n = self.domain_totals[domain]
            self.index.extend([(cursor + i, domain, i) for i in range(n)])
            cursor += n
        self.raw_len = len(self.index)

        self.dedup = dedup
        self.valid_mask = None
        if dedup:
            self.valid_mask = self._compute_dedup_mask()
        else:
            self.valid_mask = torch.ones(self.raw_len, dtype=torch.bool)

        self._build_stratified_order()

    def _compute_dedup_mask(self):
        seen = set()
        mask = torch.zeros(self.raw_len, dtype=torch.bool)
        for global_idx in range(self.raw_len):
            tokens = self._fetch_tokens(global_idx)
            h = _hash_13gram(tokens.numpy())
            if h not in seen:
                seen.add(h)
                mask[global_idx] = True
        return mask

    def _fetch_tokens(self, global_idx: int, plus_one: bool = False) -> torch.Tensor:
        _, domain, local_idx = self.index[global_idx]
        length = self.seq_len + (1 if plus_one else 0)
        for shard_path, n_seqs, start_idx in self.domain_entries[domain]:
            if local_idx < start_idx + n_seqs:
                local = local_idx - start_idx
                offset = local * self.seq_len
                # The last indexed sequence of a shard may sit exactly at the
                # file end (n_seqs = n_tokens // seq_len). Clamp the read so a
                # plus_one window never overruns the shard (mmap would raise
                # "mmap length is greater than file size").
                avail = (shard_path.stat().st_size // 2) - offset
                if length > avail:
                    length = avail
                mm = np.memmap(str(shard_path), dtype=np.uint16, mode='r',
                               offset=offset * 2, shape=(length,))
                tokens = torch.from_numpy(mm.copy().astype(np.int64))
                del mm
                return tokens
        raise IndexError(f"Bad index {global_idx}")

    def _build_stratified_order(self):
        valid_indices = torch.where(self.valid_mask)[0].tolist()
        active_domains = [d for d in self.domains if d in self.ratios]
        if not active_domains:
            active_domains = self.domains[:]
        buckets = {d: [] for d in active_domains}
        for idx in valid_indices:
            _, domain, _ = self.index[idx]
            if domain in buckets:
                buckets[domain].append(idx)
        for d in active_domains:
            random.shuffle(buckets[d])

        self.epoch_order = []
        ptrs = {d: 0 for d in active_domains}
        total_valid = len(valid_indices)

        min_ratio = min(self.ratios[d] for d in active_domains) if active_domains else 1.0
        if min_ratio <= 0:
            active_domains = [d for d in active_domains if self.ratios[d] > 0]
            if not active_domains:
                raise RuntimeError("All domain ratios are zero — cannot build epoch order")
            min_ratio = min(self.ratios[d] for d in active_domains)

        while sum(ptrs[d] < len(buckets[d]) for d in active_domains) > 0:
            for domain in active_domains:
                n_emit = max(1, round(self.ratios[domain] / min_ratio))
                for _ in range(n_emit):
                    if ptrs[domain] < len(buckets[domain]):
                        self.epoch_order.append(buckets[domain][ptrs[domain]])
                        ptrs[domain] += 1
            if len(self.epoch_order) > total_valid * 2:
                break
        self.epoch_order = self.epoch_order[:total_valid]

        jit_window = 5000
        for i in range(0, len(self.epoch_order), jit_window):
            chunk = self.epoch_order[i:i + jit_window]
            random.shuffle(chunk)
            self.epoch_order[i:i + jit_window] = chunk

    def reweight(self, ratios: dict):
        """Hot-swap: update ratios and rebuild the stratified epoch order in place."""
        self.ratios = {k: v for k, v in ratios.items() if v > 0}
        self._build_stratified_order()

    def __len__(self):
        return len(self.epoch_order)

    def __getitem__(self, idx):
        global_idx = self.epoch_order[idx]
        toks = self._fetch_tokens(global_idx, plus_one=True)  # seq_len+1 window
        x = toks[:-1].contiguous()
        y = toks[1:].contiguous()
        return {"input_ids": x, "labels": y, "domain": self.index[global_idx][1]}


def _collate_curriculum(batch):
    """Stack curriculum dicts -> (x, y, domains). Branch of dict vs plain returns."""
    if isinstance(batch[0], dict):
        x = torch.stack([b["input_ids"] for b in batch])
        y = torch.stack([b["labels"] for b in batch])
        domains = [b["domain"] for b in batch]
        return {"x": x, "y": y, "domains": domains}
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return {"x": x, "y": y, "domains": None}


def build_curriculum_dataloader(cfg, is_val=False, ratios=None):
    """Initial stratified curriculum loader. Ratios built at epoch 0; re-glided
    every CURRICULUM_UPDATE_INTERVAL steps via ds.reweight()."""
    if ratios is None:
        ratios = get_curriculum_ratios(0, cfg.max_steps)
    ds = StratifiedShardDataset(SHARD_DIRS, seq_len=cfg.seq_len,
                                ratios=ratios, dedup=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    collate_fn=_collate_curriculum, num_workers=8, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    return dl, ds


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
    """Compute next-token CE with faithful needle MTP aux loss (optional).

    Uses model.compute_loss which materializes logits in chunks (peak (B,256,V))
    instead of the full (B,T,V) — avoids the OOM wall on large B.

    Returns (loss_tensor, main_float, mtp_float):
      - loss_tensor: combined scalar tensor for backward (main + mtp_weight*mtp).
      - main_float / mtp_float: the two CE terms as Python floats, for logging.
    """
    return model.compute_loss(x, y, mtp_weight)


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
    checkpoint_dir: str = "/home/kenpeter/work/x-small/checkpoints/san"  # inside project (user rule)
    dtype: str = "bfloat16"
    curriculum: bool = False  # DoReMi-lite G1-G4 stratified dynamic mix


def _compile_model(model):
    """Compile for max throughput (Triton-backed inductor). Falls back gracefully.

    Mode chain: default -> max-autotune-no-cudagraphs -> eager.
    CUDA-graphs deliberately OFF: profile shows the step is GPU-bound (720ms GPU
    vs 539ms CPU) — graphs only save CPU launch time — and grad_accum's 8
    forwards-then-backwards pattern corrupts the captured pool (stale RoPE cos
    reads). "default" mode = fast compile (no 13-min autotune tax); the Triton
    kernels + thousands of tiny elementwise kernels dominate, so mode
    max-autotune buys little extra.
    """
    # Compile the ACTUAL hot path (san_loss -> model.compute_loss -> _hidden),
    # NOT model.forward (which training never calls — that's why the old
    # `torch.compile(model)` gave zero speedup + a 13-min autotune tax).
    # dynamic=False: B,T,L shapes are fixed -> no recompiles, fastest path.
    for mode in ("default", "max-autotune-no-cudagraphs"):
        try:
            model.compute_loss = torch.compile(model.compute_loss,
                                               dynamic=False, mode=mode)
            print(f"  ✅ compiled compute_loss mode={mode}")
            return model
        except Exception as e:
            print(f"  ⚠ torch.compile(compute_loss,{mode}) failed: {e}; trying next")
    print("  ⚠ torch.compile failed; running eager (no compile)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--mtp-weight", type=float, default=0.1)
    ap.add_argument("--no-compile", action="store_true",
                   help="run eager (skip torch.compile) for instant startup")
    ap.add_argument("--log-every", type=int, default=10,
                   help="log a training line every N steps")
    ap.add_argument("--data-dir", default="/home/kenpeter/work/data/_shards_final")
    ap.add_argument("--checkpoint-dir", default="/home/kenpeter/work/checkpoints/xsmall_san")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume")
    ap.add_argument("--curriculum", action="store_true",
                    help="use DoReMi-lite G1-G4 stratified dynamic mix instead of flat farm")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="disable grad-checkpointing (MHC layers) — faster compute, more VRAM")
    args = ap.parse_args()

    cfg = TrainConfig(
        max_steps=args.steps, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, seq_len=args.seq_len, mtp_weight=args.mtp_weight,
        data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every, curriculum=args.curriculum,
        use_checkpoint=not args.no_checkpoint, log_every=args.log_every)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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
                                  weight_decay=cfg.weight_decay, fused=True)

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
    if getattr(args, "no_compile", False):
        print("  ⚡ eager mode (no torch.compile)")
    else:
        model = _compile_model(model)
    train_iter = iter(train_loader)
    n_shards = len(ds.shards) if hasattr(ds, "shards") else len(ds.domains)
    print(f"Data: {'curriculum' if cfg.curriculum else 'flat farm'}  sources={n_shards}  "
          f"steps={cfg.max_steps}  eff_batch={cfg.batch_size*cfg.grad_accum}")

    next_glide = CURRICULUM_UPDATE_INTERVAL if cfg.curriculum else None

    import signal as _signal
    _stop = {"v": False}
    def _on_signal(signum, frame):
        print(f"\n  ⏸ signal {signum} — will save checkpoint and exit at next step boundary")
        _stop["v"] = True
    _signal.signal(_signal.SIGTERM, _on_signal)
    _signal.signal(_signal.SIGINT, _on_signal)

    def save_checkpoint(step):
        state = {"step": step, "loss": acc_loss,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "config": cfg.__dict__}
        tmp = latest_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, latest_path)
        print(f"  💾 saved latest (step {step}) -> {latest_path}")

    t0 = time.time()
    running = 0.0
    step = start_step
    while step < cfg.max_steps:
        step += 1

        if _stop["v"]:
            save_checkpoint(step)
            break

        # DoReMi re-glide: recompute ratios (hot-reloads curriculum_boost.json
        # each call) and hot-swap the epoch order every CURRICULUM_UPDATE_INTERVAL.
        if cfg.curriculum and step >= next_glide:
            new_ratios = get_curriculum_ratios(step, cfg.max_steps)
            ds.reweight(new_ratios)
            train_iter = iter(train_loader)
            print(f"  🔄 curriculum re-glide @ step {step}: "
                  f"{sum(1 for v in new_ratios.values() if v > 0)} active domains, "
                  f"epoch_order={len(ds)}")
            next_glide = step + CURRICULUM_UPDATE_INTERVAL

        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        acc_loss = 0.0
        acc_main, acc_mtp = 0.0, 0.0
        acc_dom_loss, acc_dom_n = {}, {}
        for _ in range(cfg.grad_accum):
            # CUDA-graph safety: each grad-accum invocation reuses the captured
            # graph region; mark the step so stale graph-output reads (e.g. RoPE
            # cos/sin saved for backward) can't hit an overwritten pool. No-op
            # when cudagraphs are disabled.
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            if isinstance(batch, dict):  # curriculum path
                x, y = batch["x"], batch["y"]
            else:                        # flat farm path
                x, y = batch
            x, y = x.to(device), y.to(device)
            out = san_loss(model, x, y, cfg.mtp_weight)
            loss, main_ce, mtp_ce = out  # loss=tensor (backward), main/mtp=floats
            loss = loss / cfg.grad_accum
            acc_loss += loss.item() * cfg.grad_accum
            acc_main += (main_ce if main_ce else 0.0)
            acc_mtp += (mtp_ce if mtp_ce else 0.0)
            if isinstance(batch, dict) and batch.get("domains") is not None:
                _dom = batch["domains"][0]
                acc_dom_loss[_dom] = acc_dom_loss.get(_dom, 0.0) + loss.item() * cfg.grad_accum
                acc_dom_n[_dom] = acc_dom_n.get(_dom, 0) + 1
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += acc_loss

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_s = (cfg.batch_size * cfg.seq_len * cfg.grad_accum * cfg.log_every) / dt
            n = cfg.log_every
            ga = cfg.grad_accum
            print(f"step {step:6d} | loss {running/n/ga:.4f} "
                  f"[main {acc_main/ga:.3f} | mtp {acc_mtp/ga:.3f}] "
                  f"lr {lr:.2e} | {dt:.1f}s | {tok_s:,.0f} tok/s")
            if acc_dom_loss:
                _pd = " · ".join(f"{d} {acc_dom_loss[d]/acc_dom_n[d]:.2f}"
                                 for d in sorted(acc_dom_loss))
                print(f"  per-domain: {_pd}")
            running, acc_main, acc_mtp = 0.0, 0.0, 0.0
            acc_dom_loss, acc_dom_n = {}, {}

        if step % cfg.save_every == 0 or step == cfg.max_steps:
            save_checkpoint(step)

    print("✅ done")


if __name__ == "__main__":
    main()
