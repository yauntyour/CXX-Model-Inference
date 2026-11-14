"""AstrAI-V1-2B-instruct PyTorch reference (validation only).

Reads the same safetensors file as the C++ engine and runs a plain
Llama-style forward pass, so token IDs / logits / greedy continuations can
be compared against astrai_chat.

Usage:
  python tools/astrai_reference.py --model D:/models/AstrAI-V1-2B-instruct \
         --prompt "Hello" --max-new 20
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer


def rms_norm(x, w, eps, one_plus_w=False):
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return y * (1.0 + w if one_plus_w else w)


def rotate_half(x, rot):
    x1 = x[..., : rot // 2]
    x2 = x[..., rot // 2 : rot]
    return torch.cat((-x2, x1), dim=-1)


class Reference:
    def __init__(self, model_dir, device="cuda", rope_interleaved=None, norm_1pw=False,
                 mlp_act="silu"):
        self.device = device
        cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.cfg = cfg
        if rope_interleaved is None:
            if "attention" in cfg and "gqa" in cfg["attention"]:
                rope_interleaved = cfg["attention"]["gqa"].get("rope_interleaved", True)
            else:
                rope_interleaved = cfg.get("rope_interleaved", False)
        self.rope_interleaved = rope_interleaved
        self.norm_1pw = norm_1pw
        self.mlp_act = mlp_act
        att = cfg.get("attention", {})
        gqa = att.get("gqa", {})
        self.H = cfg["hidden_size"]
        self.I = cfg["intermediate_size"]
        self.V = cfg["vocab_size"]
        self.L = cfg["num_hidden_layers"]
        self.Hq = att.get("num_heads", cfg.get("num_attention_heads"))
        self.Hkv = att.get("num_kv_heads", cfg.get("num_key_value_heads", self.Hq))
        self.D = gqa.get("head_dim", cfg.get("head_dim", self.H // self.Hq))
        self.rot = gqa.get("rotary_dim", cfg.get("rotary_dim", self.D))
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.theta = cfg.get("rope_theta", 10000.0)

        path = os.path.join(model_dir, "model.safetensors")
        self.w = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k).to(device)
        self.embed = self.w["model.embed_tokens.weight"]
        self.lm_head = self.w.get("lm_head.weight", self.embed)
        self.norm = self.w["model.norm.weight"]

    def _attn(self, h, li, base, kcache=None, vcache=None):
        p = f"model.layers.{li}.attention."
        q = (h @ self.w[p + "q_proj.weight"].T).view(-1, self.Hq, self.D)
        k = (h @ self.w[p + "k_proj.weight"].T).view(-1, self.Hkv, self.D)
        v = (h @ self.w[p + "v_proj.weight"].T).view(-1, self.Hkv, self.D)

        S = q.shape[0]
        pos = torch.arange(base, base + S, device=q.device, dtype=torch.float32)
        half = self.rot // 2
        inv = 1.0 / (self.theta ** (torch.arange(half, device=q.device, dtype=torch.float32) / half))
        emb = pos[:, None] * inv[None, :]
        cos = torch.cat([emb.cos(), emb.cos()], dim=-1)[:, None, :]
        sin = torch.cat([emb.sin(), emb.sin()], dim=-1)[:, None, :]
        cos_h = emb.cos()[:, None, :]
        sin_h = emb.sin()[:, None, :]

        def rope(x):
            xr = x[..., : self.rot].float()
            if self.rope_interleaved:
                x1 = xr[..., 0::2]
                x2 = xr[..., 1::2]
                out = torch.stack(
                    (x1 * cos_h - x2 * sin_h, x1 * sin_h + x2 * cos_h), dim=-1
                ).flatten(-2).to(x.dtype)
            else:
                out = (xr * cos + rotate_half(xr, self.rot) * sin).to(x.dtype)
            if self.rot < self.D:
                out = torch.cat([out, x[..., self.rot :]], dim=-1)
            return out

        q = rope(q)
        k = rope(k)

        if kcache is not None:
            kcache[base : base + S] = k
            vcache[base : base + S] = v
            k = kcache[: base + S]
            v = vcache[: base + S]

        k = k.repeat_interleave(self.Hq // self.Hkv, dim=1)
        v = v.repeat_interleave(self.Hq // self.Hkv, dim=1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * (1.0 / (self.D ** 0.5))
        rows = torch.arange(S, device=q.device)[:, None] + base
        cols = torch.arange(base + S, device=q.device)[None, :]
        scores = scores.masked_fill(cols > rows, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        o = (probs @ v).transpose(0, 1).reshape(S, self.Hq * self.D)
        return o @ self.w[p + "o_proj.weight"].T

    def forward(self, ids, caches=None, all_positions=False):
        x = self.embed[torch.tensor(ids, device=self.device)]
        base = 0 if caches is None else caches["len"]
        for li in range(self.L):
            p = f"model.layers.{li}."
            h = rms_norm(x, self.w[p + "input_norm.weight"], self.eps, self.norm_1pw)
            kc = None if caches is None else caches["k"][li]
            vc = None if caches is None else caches["v"][li]
            x = x + self._attn(h, li, base, kc, vc)
            h = rms_norm(x, self.w[p + "post_attention_norm.weight"], self.eps, self.norm_1pw)
            g = (h @ self.w[p + "mlp.gate.weight"].T)
            u = (h @ self.w[p + "mlp.up.weight"].T)
            if self.mlp_act == "gelu":
                g = F.gelu(g)
            elif self.mlp_act == "gelu_tanh":
                g = F.gelu(g, approximate="tanh")
            else:
                g = F.silu(g)
            x = x + ((g * u) @ self.w[p + "mlp.down.weight"].T)
        h = rms_norm(x, self.norm, self.eps, self.norm_1pw)
        if all_positions:
            logits = h.float() @ self.lm_head.float().T
        else:
            logits = h[-1].float() @ self.lm_head.float().T
        if caches is not None:
            caches["len"] += len(ids)
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true",
                    help="每步从完整序列重算 (消除 cache 影响, 便于对比)")
    ap.add_argument("--rope-interleaved", action="store_true")
    ap.add_argument("--rope-neox", action="store_true")
    ap.add_argument("--norm-1pw", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    print("prompt ids:", ids)

    ri = False if args.rope_neox else (True if args.rope_interleaved else None)
    ref = Reference(args.model, rope_interleaved=ri, norm_1pw=args.norm_1pw)
    logits = ref.forward(ids)
    top = torch.topk(logits, args.topk)
    print("top ids :", top.indices.tolist())
    print("top vals:", [round(v, 4) for v in top.values.tolist()])
    print("next:", tok.decode([int(top.indices[0])]))

    gen = []
    toks = list(ids)
    caches = None
    if not args.no_cache:
        max_len = len(ids) + args.max_new + 8
        caches = {
            "len": 0,
            "k": [torch.empty(max_len, ref.Hkv, ref.D, device=ref.device,
                              dtype=torch.bfloat16) for _ in range(ref.L)],
            "v": [torch.empty(max_len, ref.Hkv, ref.D, device=ref.device,
                              dtype=torch.bfloat16) for _ in range(ref.L)],
        }
        logits = ref.forward(ids, caches)

    for _ in range(args.max_new):
        if caches is None:
            lg = ref.forward(toks)
        else:
            lg = ref.forward([toks[-1]], caches) if gen else logits
        nxt = int(torch.argmax(lg))
        if nxt in (1, 130817):
            print("hit eos", nxt)
            break
        gen.append(nxt)
        toks.append(nxt)
        print(tok.decode([nxt]), end="", flush=True)
    print()
    print("generated ids:", gen)
    print("generated text:", repr(tok.decode(gen)))


if __name__ == "__main__":
    sys.exit(main())
