"""Qualitative + quantitative eval of the trained SAN (x-small) model.

Loads checkpoints/san/san_latest.pt, runs GREEDY generation on real prompts
(knowledge / code / math / reasoning) and computes perplexity on real held-out
passages. Saves full output to evaluations/san_eval_step<step>_<ts>.txt.

Usage: ./venv_xsmall/bin/python eval_san_gen.py
"""
import os
import sys
import datetime
import textwrap
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from san_model import SimpleAttentionNetwork, SANConfig

CKPT = "checkpoints/san/san_latest.pt"
TOK = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = 120
EVAL_MAXLEN = 512  # cap perplexity passages


def build_cfg(d):
    return SANConfig(
        vocab_size=d["vocab_size"],
        d_model=d["d_model"],
        num_heads=d["num_heads"],
        num_kv_heads=d["num_kv_heads"],
        num_layers=d["num_layers"],
        max_seq_len=d["seq_len"],
        use_checkpoint=d["use_checkpoint"],
        dtype=d["dtype"],
        ckpt_every=3,
    )


def greedy_generate(model, tok, prompt, max_new=MAX_NEW, rep_penalty=1.2):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    gen = ids.clone()
    eos = tok.eos_token_id
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(gen)[:, -1, :]
            if rep_penalty != 1.0:
                # repetition penalty: suppress already-emitted tokens
                for t in set(gen[0].tolist()):
                    logits[0, t] /= rep_penalty
            nxt = logits.argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=-1)
            if eos is not None and nxt.item() == eos:
                break
    return tok.decode(gen[0], skip_special_tokens=True)


def perplexity(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    if ids.shape[1] < 2:
        return None, None
    if ids.shape[1] > EVAL_MAXLEN:
        ids = ids[:, :EVAL_MAXLEN]
    with torch.no_grad():
        logits = model(ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        ids[:, 1:].reshape(-1),
    )
    return float(loss), float(torch.exp(loss))


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

PASSAGES = [
    ("prose", "The quick brown fox jumps over the lazy dog. Machine learning models learn patterns from large amounts of training data."),
    ("code",  "def add(a, b):\n    return a + b\n\nresult = add(3, 4)\nprint(result)"),
    ("math",  "The integral of x squared is x cubed over three. The derivative measures the rate of change of a function with respect to its variable."),
]


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    step = ck["step"]
    train_loss = ck.get("loss")
    cfg = build_cfg(ck["config"])
    model = SimpleAttentionNetwork(cfg).to(DEVICE)
    missing, unexpected = [], []
    sd = ck["model_state_dict"]
    cur = model.state_dict()
    missing = [k for k in cur if k not in sd]
    unexpected = [k for k in sd if k not in cur]
    model.load_state_dict(sd, strict=True)
    model.eval()
    nparams = model.count_parameters() / 1e6
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)

    L = []
    L.append(f"SAN (x-small) EVAL  |  step={step}  params={nparams:.2f}M  device={DEVICE}")
    L.append(f"training loss @ckpt = {train_loss}")
    L.append(f"state_dict check: missing={len(missing)} unexpected={len(unexpected)}")
    L.append("=" * 72)

    L.append("GREEDY GENERATION (real prompts):")
    for dom, p in PROMPTS:
        txt = greedy_generate(model, tok, p)
        L.append(f"\n### [{dom}] prompt: {p!r}")
        L.append(textwrap.indent(txt, "    "))

    L.append("\n" + "=" * 72)
    L.append("PERPLEXITY on real passages (lower = better):")
    agg = []
    for dom, txt in PASSAGES:
        loss, ppl = perplexity(model, tok, txt)
        if loss is None:
            continue
        agg.append(ppl)
        L.append(f"  [{dom:8s}] loss={loss:.3f}  ppl={ppl:.2f}")
    if agg:
        L.append(f"  mean ppl = {sum(agg)/len(agg):.2f}")

    full = "\n".join(L)
    os.makedirs("evaluations", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fpath = f"evaluations/san_eval_step{step}_{ts}.txt"
    with open(fpath, "w") as f:
        f.write(full + "\n")
    print(full)
    print("\nSAVED ->", fpath)


if __name__ == "__main__":
    main()
