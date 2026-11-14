"""Run astrai_chat --bench over several context lengths and print a table."""

import argparse
import os
import re
import shlex
import subprocess

UNIT = (
    "The quick brown fox jumps over the lazy dog. Artificial intelligence is "
    "transforming the way people work, learn, and communicate with each other. "
    "In a distant galaxy, a small spacecraft drifted silently through the darkness "
    "while its crew slept. The history of computing is full of unexpected turns, "
    "from room-sized machines to devices that fit in a pocket. "
)


def run(exe, model, prompt, max_new, extra, env_extra):
    env = os.environ.copy()
    env.update(env_extra)
    p = subprocess.run(
        [exe, "--model", model, "--bench", str(max_new), "--probe", prompt] + extra,
        capture_output=True, text=True, encoding="utf-8", timeout=3600, env=env,
    )
    out = p.stdout
    m = re.search(r"prompt=(\d+) tokens", out)
    pre = re.search(r"prefill:\s*([\d.]+) ms,\s*([\d.]+) tok/s", out)
    dec = re.search(r"decode\s*:\s*([\d.]+) ms,\s*([\d.]+) tok/s", out)
    return (int(m.group(1)) if m else -1,
            float(pre.group(2)) if pre else 0.0,
            float(dec.group(2)) if dec else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=r"build-astrai\Release\astrai_chat.exe")
    ap.add_argument("--model", default=r"D:\models\AstrAI-V1-2B-instruct")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--reps", default="1,6,14")
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", default="")
    args = ap.parse_args()

    extra = shlex.split(args.extra)
    env_extra = {}
    for kv in filter(None, args.env.split(",")):
        k, v = kv.split("=", 1)
        env_extra[k] = v

    reps_list = [int(x) for x in args.reps.split(",")]
    print(f"{'reps':>4} {'tokens':>7} {'prefill tok/s':>14} {'decode tok/s':>13}")
    for reps in reps_list:
        prompt = UNIT * reps
        n, pre, dec = run(args.exe, args.model, prompt, args.max_new, extra, env_extra)
        print(f"{reps:>4} {n:>7} {pre:>14.0f} {dec:>13.1f}")


if __name__ == "__main__":
    main()
