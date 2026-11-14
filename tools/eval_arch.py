"""Compare architecture variants by mean token NLL on fixed texts.

The correct RoPE pairing / RMSNorm form should give a clearly lower NLL.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astrai_reference import Reference  # noqa: E402

TEXTS = [
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains every letter of the alphabet.",
    "Artificial intelligence is transforming the way people work, learn, "
    "and communicate with each other.",
    "你好，今天天气很好，我们一起去公园散步吧。",
    "def fibonacci(n):\n    if n < 2:\n        return n\n"
    "    return fibonacci(n-1) + fibonacci(n-2)\n",
    "In a distant galaxy, a small spacecraft drifted silently through the "
    "darkness, its crew asleep.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    enc = [tok.encode(t, add_special_tokens=False).ids for t in TEXTS]
    ref = Reference(args.model)

    variants = [
        ("neox  + silu", False, False, "silu"),
        ("inter + silu", True, False, "silu"),
        ("inter + gelu", True, False, "gelu"),
        ("inter + gelu_tanh", True, False, "gelu_tanh"),
    ]
    for name, inter, one, act in variants:
        ref.rope_interleaved = inter
        ref.norm_1pw = one
        ref.mlp_act = act
        total, count = 0.0, 0
        per = []
        for ids in enc:
            logits = ref.forward(ids, all_positions=True)
            logp = F.log_softmax(logits[:-1].float(), dim=-1)
            tgt = torch.tensor(ids[1:], device=logits.device)
            nll = -logp[torch.arange(len(tgt)), tgt]
            total += nll.sum().item()
            count += len(tgt)
            per.append(nll.mean().item())
        print(f"{name:26s} mean NLL = {total / count:.4f}   "
              f"[{', '.join(f'{x:.3f}' for x in per)}]")


if __name__ == "__main__":
    sys.exit(main())
