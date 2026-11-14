"""Allow the AstrAI repo (requires torch 2.13) to import under torch 2.8.

Only inference is used; the Muon optimizer internals are never called.
"""

import sys
import types

try:
    import torch  # noqa: F401
except Exception:
    raise

if "torch.optim._muon" not in sys.modules:
    mod = types.ModuleType("torch.optim._muon")

    def _adjust_lr(*args, **kwargs):
        raise RuntimeError("torch.optim._muon is a stub (inference only)")

    def _single_tensor_muon(*args, **kwargs):
        raise RuntimeError("torch.optim._muon is a stub (inference only)")

    def _zeropower_via_newtonschulz(*args, **kwargs):
        raise RuntimeError("torch.optim._muon is a stub (inference only)")

    mod._adjust_lr = _adjust_lr
    mod._single_tensor_muon = _single_tensor_muon
    mod._zeropower_via_newtonschulz = _zeropower_via_newtonschulz
    sys.modules["torch.optim._muon"] = mod

if not hasattr(torch.optim, "Muon"):
    class _MuonStub(torch.optim.Optimizer):
        def __init__(self, params, **kwargs):
            super().__init__(params, {})

        def step(self, closure=None):
            raise RuntimeError("torch.optim.Muon is a stub (inference only)")

    torch.optim.Muon = _MuonStub
