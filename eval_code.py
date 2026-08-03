"""Coding-focused eval of latest pretrained checkpoint — greedy generations on code prompts."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import SmolLM2, TrainConfig
from transformers import AutoTokenizer

CKPT = "/home/kenpeter/work/checkpoints/pretrained_latest.pt"
DEVICE = "cuda"

PROMPTS = [
    "def add(a, b):\n    ",
    "def multiply(a, b):\n    ",
    "def factorial(n):\n    ",
    "def fibonacci(n):\n    ",
    "def is_even(x):\n    ",
    "def greet(name):\n    ",
    "def bubble_sort(arr):\n    ",
    "for i in range(5):\n    ",
    "class Dog:\n    ",
    "x = [1, 2, 3]\nfor ",
    "SELECT * FROM users WHERE",
    "#!/usr/bin/env python3\nimport ",
]

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg_kwargs = {k: v for k, v in ckpt.get("config", {}).items()
              if k in TrainConfig.__dataclass_fields__}
cfg = TrainConfig(**cfg_kwargs)
print(f"Loaded checkpoint: step {ckpt.get('step')}, best_loss {ckpt.get('best_loss', float('nan')):.4f}")

model = SmolLM2(cfg).to(DEVICE)
sd = ckpt["model_state_dict"]
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()
del ckpt

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

@torch.no_grad()
def gen(prompt, max_new=30):
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_new):
        logits, _ = model(inp)
        nxt = logits[0, -1, :].argmax(dim=-1, keepdim=True)
        inp = torch.cat([inp, nxt.unsqueeze(0)], dim=1)
    return tok.decode(inp[0].tolist())

for p in PROMPTS:
    out = gen(p)
    print(f"\n🔹 {p!r}\n   → {out[len(p):]!r}")
