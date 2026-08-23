"""Throughput ceiling sweep for x-small / SAN.

Faithfully replicates train_san.san_loss (main + 0.1*MTP aux) so tok/s reflects
the real training compute path. use_checkpoint=False (grad-ckpt ON trades compute
for VRAM — off here because 45M on 12GB has VRAM to spare, giving the compute
ceiling). Compiles ONCE with dynamic=True so the sweep doesn't recompile per shape.
"""
import time, torch, argparse
from san_model import SimpleAttentionNetwork, SANConfig
from train_san import san_loss

device = "cuda"
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

cfg = SANConfig(vocab_size=49152, d_model=384, num_heads=8, num_kv_heads=4,
                num_layers=27, max_seq_len=4096, use_checkpoint=False, dtype="bfloat16")
model = SimpleAttentionNetwork(cfg).to(device).train()
print(f"PARAMS: {model.count_parameters()/1e6:.2f}M", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=4e-4, fused=True)
print("Compiling (dynamic) ...", flush=True)
model = torch.compile(model, dynamic=True, mode="max-autotune-no-cudagraphs")
print("Compiled.", flush=True)

MTP = 0.1
VOCAB = 49152

def run(B, S, warm=4, steps=12):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(warm + steps):
        x = torch.randint(0, VOCAB, (B, S), device=device)
        y = torch.randint(0, VOCAB, (B, S), device=device)
        loss, _, _ = san_loss(model, x, y, MTP)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if i == warm - 1:
            torch.cuda.synchronize(); t0 = time.time()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return (B * S * steps) / dt, torch.cuda.max_memory_allocated() / 1e9

GRID = [(B, S) for B in (8, 16, 32, 64, 96, 128) for S in (1024, 2048, 4096)]
print(f"{'B':>5} {'S':>6} {'tok/s':>10} {'peakVRAM':>9}", flush=True)
best = (0, 0, 0)
for B, S in GRID:
    try:
        tps, mem = run(B, S)
        print(f"{B:>5} {S:>6} {tps:>10,.0f} {mem:>8.1f}GB", flush=True)
        if tps > best[0]:
            best = (tps, B, S)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{B:>5} {S:>6} {'OOM':>10}", flush=True)
print(f"\nCEILING: {best[0]:,.0f} tok/s @ B={best[1]} S={best[2]}", flush=True)
