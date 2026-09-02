#!/usr/bin/env python3
"""Temp-sampled generation eval for the SAN (x-small) checkpoint.

Loads checkpoints/san/san_latest.pt, runs stochastic decoding (temp/top-k/top-p)
on the same real prompts as eval_san_gen.py. Small models often escape greedy
repetition loops under sampling, giving a fairer 'is it good' read.
Saves output to evaluations/san_sample_step<step>_<ts>.txt
"""
import os, math, time, datetime
from dataclasses import fields
import torch
from transformers import AutoTokenizer

CKPT = "checkpoints/san/san_latest.pt"
TOK  = "HuggingFaceTB/SmolLM2-135M"
MAX_NEW = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMP   = 0.8
TOP_K  = 40
TOP_P  = 0.9
REP_PENALTY = 1.8  # suppress already-generated tokens to curb repetition loops
NO_REPEAT_NGRAM = 4  # forbid next token if it would recreate an existing 4-gram

PROMPTS = [
    ("knowledge", "The capital of France is"),
    ("knowledge", "Albert Einstein is best known for"),
    ("code-py",   "def quicksort(arr):"),
    ("code-py",   "import numpy as np\n\ndef moving_average(x, window):"),
    ("code-js",   "// JavaScript: fetch JSON from an API and log it\n"),
    ("math",      "Solve for x: 2x + 5 = 13. The answer is x ="),
    ("reasoning", "The opposite of 'happy' is"),
    ("open",      "Write a short poem about the ocean:"),
]


def sample(logits, temp, top_k, top_p):
    if temp == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temp
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, -1:]] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits[remove.scatter(-1, sorted_idx, remove)] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    from san_model import SimpleAttentionNetwork, SANConfig  # noqa
    ckpt = torch.load(CKPT, map_location="cpu")
    cfg_dict = ckpt["config"]
    if isinstance(cfg_dict, SANConfig):
        cfg = cfg_dict
    else:
        valid = {f.name for f in fields(SANConfig)}
        filtered = {}
        for k, v in cfg_dict.items():
            if k == "seq_len":
                filtered["max_seq_len"] = v
            elif k in valid:
                filtered[k] = v
        cfg = SANConfig(**filtered)
    model = SimpleAttentionNetwork(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = AutoTokenizer.from_pretrained(TOK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    step = ckpt.get("step", -1)
    n_params = sum(p.numel() for p in model.parameters())
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"evaluations/san_sample_step{step}_{ts}.txt"
    os.makedirs("evaluations", exist_ok=True)

    lines = []
    lines.append(f"SAN (x-small) SAMPLED EVAL | step={step} params={n_params/1e6:.2f}M device={DEVICE}")
    lines.append(f"decode: temp={TEMP} top_k={TOP_K} top_p={TOP_P} rep_pen={REP_PENALTY} no_repeat_ngram={NO_REPEAT_NGRAM} max_new={MAX_NEW}")
    lines.append(f"training loss @ckpt = {ckpt.get('loss')}")
    lines.append("=" * 72)

    torch.manual_seed(0)
    with torch.no_grad():
        for tag, prompt in PROMPTS:
            ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
            gen = ids.clone()
            for _ in range(MAX_NEW):
                logits = model(gen)[:, -1, :]
                if REP_PENALTY != 1.0:
                    for t in set(gen[0].tolist()):
                        logits[0, t] /= REP_PENALTY
                # no-repeat n-gram: ban next token if it would recreate an existing n-gram
                if NO_REPEAT_NGRAM > 0 and gen.size(1) >= NO_REPEAT_NGRAM:
                    n = NO_REPEAT_NGRAM
                    prev = gen[0, -(n - 1):].tolist()
                    ngrams = {tuple(gen[0, i:i + n].tolist())
                              for i in range(gen.size(1) - n + 1)}
                    for tid in range(logits.size(-1)):
                        if tuple(prev + [tid]) in ngrams:
                            logits[0, tid] = -float("inf")
                nxt = sample(logits, TEMP, TOP_K, TOP_P)
                gen = torch.cat([gen, nxt], dim=-1)
                if nxt.item() == tok.eos_token_id:
                    break
            text = tok.decode(gen[0][ids.size(1):], skip_special_tokens=True)
            lines.append(f"\n### [{tag}] prompt: {prompt!r}")
            lines.append(text)

    lines.append("=" * 72)
    lines.append(f"SAVED -> {out_path}")
    out = "\n".join(lines)
    print(out)
    with open(out_path, "w") as f:
        f.write(out)


if __name__ == "__main__":
    main()
