"""Runtime fix for TE 2.17.0+miles -> FA4 4.0.0b35 on Blackwell (miles v0.1.3 image). Loaded via PYTHONPATH.

TransformerEngine passes window_size=(-1, 0) for plain causal attention. FA4's _resolve_causal_local_window only
treats None (or both sides negative) as "no window", so (-1, 0) turns into *local* attention with left window -1:
the forward is wrong (trainer logprobs -1.87 vs SGLang -0.39 on Qwen3.5-0.8B) and the head_dim=256 SM100 backward
asserts "does not support local attention". Here a negative window side is mapped to None (unbounded), which is
what TE means by -1. Nothing else changes. Installed as a post-import hook so flash_attn.cute is not imported
eagerly in every process.
"""

import importlib.abc
import importlib.util
import sys


_TARGET = "flash_attn.cute.interface"


def _patch(mod):
    orig = mod._resolve_causal_local_window

    def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
        if window_size_left is not None and window_size_left < 0:
            window_size_left = None
        if window_size_right is not None and window_size_right < 0:
            window_size_right = None
        return orig(causal, window_size_left, window_size_right, mask_mod)

    mod._resolve_causal_local_window = _resolve_causal_local_window
    mod._miles_bench_fa4_window_fix = True


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _Finder())
