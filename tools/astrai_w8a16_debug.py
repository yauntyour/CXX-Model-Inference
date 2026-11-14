import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "AstrAI"
MODEL = Path(r"D:\models\AstrAI-V1-2B-instruct")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "tools"))

_spec = importlib.util.spec_from_file_location(
    "astrai_w8a16_mod",
    ROOT / "tools" / "astrai_w8a16_gemv.py",
)
extmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extmod)
from astrai.config import ConfigFactory
from astrai.model import AutoModel
from astrai.serialization import adapt_config


def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


raw = json.loads((MODEL / "config.json").read_text(encoding="utf-8-sig"))
config = ConfigFactory.load(adapt_config(raw, str(MODEL)))
model = AutoModel.from_pretrained(str(MODEL))
model.to("cuda", dtype=torch.bfloat16)
model.eval()

picked = {}
for name, mod in model.named_modules():
    w = getattr(mod, "weight", None)
    if isinstance(w, torch.nn.Parameter) and w.dim() == 2:
        key = tuple(w.shape)
        if key not in picked:
            picked[key] = (name, w)

for shape, (name, w) in sorted(picked.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
    x = torch.randn(1, w.shape[1], device="cuda", dtype=torch.bfloat16)
    tq = bench(lambda: extmod._qlinear(x, w))
    tb = bench(lambda: extmod._orig_linear(x, w))
    print(f"{name} {tuple(w.shape)}: w8a16 {tq:.3f} ms, bf16 {tb:.3f} ms, ratio {tb/tq:.2f}x")
print("cache entries", len(extmod._qcache))
