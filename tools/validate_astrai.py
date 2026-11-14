"""Compare astrai_chat (C++/CUDA) with the PyTorch reference.

Runs a raw prompt through both, compares the next-token top-k and the greedy
continuation. Requires the model.safetensors weights and tokenizer.json.

Usage:
  python tools/validate_astrai.py --model D:/models/AstrAI-V1-2B-instruct \
         --prompt "Hello" --max-new 20
"""

import argparse
import os
import re
import subprocess
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astrai_reference import Reference  # noqa: E402


def parse_top10(text):
    m = re.search(r"\[logits top10\]:\s*(.*)", text)
    if not m:
        return []
    return [int(x) for x in re.findall(r"(\d+)\(", m.group(1))]


def parse_gen_ids(text):
    m = re.search(r"\[probe\] gen ids:\s*([0-9\s]+)", text)
    if not m:
        return []
    return [int(x) for x in m.group(1).split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--exe", default=r"build-astrai\Release\astrai_chat.exe")
    ap.add_argument("--rope-interleaved", action="store_true")
    ap.add_argument("--rope-neox", action="store_true")
    ap.add_argument("--norm-1pw", action="store_true")
    ap.add_argument("--bf16", action="store_true",
                    help="关闭 W8A16，用 BF16 权重与 Python 逐 token 对拍")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print("missing exe:", exe)
        return 2

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    print("prompt ids:", ids)

    cmd = [exe, "--model", args.model, "--probe", args.prompt,
           "--max-new", str(args.max_new)]
    if args.rope_interleaved:
        cmd.append("--rope-interleaved")
    if args.rope_neox:
        cmd.append("--rope-neox")
    if args.norm_1pw:
        cmd.append("--norm-1pw")
    if args.bf16:
        cmd.append("--bf16")
    print("running cpp...")
    cpp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if cpp.returncode != 0:
        print(cpp.stdout)
        print(cpp.stderr)
        return cpp.returncode
    cpp_top = parse_top10(cpp.stdout)
    cpp_gen = parse_gen_ids(cpp.stdout)
    print("cpp top10:", cpp_top)
    print("cpp gen  :", cpp_gen)

    print("running python reference...")
    ri = False if args.rope_neox else (True if args.rope_interleaved else None)
    ref = Reference(args.model, rope_interleaved=ri, norm_1pw=args.norm_1pw)
    logits = ref.forward(ids)
    py_top = torch.topk(logits, 10).indices.tolist()
    print("py  top10:", py_top)

    py_gen = []
    toks = list(ids)
    for _ in range(args.max_new):
        lg = ref.forward(toks)
        nxt = int(torch.argmax(lg))
        if nxt in (1, 130817):
            break
        py_gen.append(nxt)
        toks.append(nxt)
    print("py  gen  :", py_gen)

    ok_top = bool(cpp_top) and cpp_top[0] == py_top[0]
    overlap = len(set(cpp_top[:5]) & set(py_top[:5]))
    ok_gen = cpp_gen == py_gen
    print()
    print("top1 match:", ok_top, "| top5 overlap:", overlap, "/5")
    print("greedy match:", ok_gen)
    if ok_gen:
        print("text:", repr(tok.decode(py_gen)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
