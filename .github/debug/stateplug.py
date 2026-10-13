# Per-test probe of thread-local state (encoding, FP mode) and newly loaded native libraries
import ctypes, locale, os, threading

_libm = ctypes.CDLL("libm.so.6")


def _state():
    env = (ctypes.c_ubyte * 32)()
    _libm.fegetenv(env)
    mxcsr = int.from_bytes(bytes(env[28:32]), "little")
    return f"enc={locale.nl_langinfo(locale.CODESET)} round={_libm.fegetround()} mxcsr={mxcsr & 0xffc0:#06x} glob={locale.setlocale(locale.LC_CTYPE)} threads={threading.active_count()}"


def _libs():
    with open("/proc/self/maps") as f:
        return {line.split()[-1] for line in f if ".so" in line}


_last, _last_libs = None, set()


def pytest_configure(config):
    import torch
    from transformers.models.llava_next import modeling_llava_next

    if os.environ.get("DISABLE_CUDNN") == "1":
        torch.backends.cudnn.enabled = False

    original = modeling_llava_next.LlavaNextModel.pack_image_features

    def pack_image_features(self, image_features, image_sizes, *args, **kwargs):
        out = original(self, image_features, image_sizes, *args, **kwargs)
        shapes = [tuple(f.shape) for f in image_features]
        _log(f"   llava_next image_sizes={image_sizes.tolist()} ({image_sizes.device}, {image_sizes.dtype}) features={shapes} -> feature_lens={out[1].tolist()}")
        return out

    modeling_llava_next.LlavaNextModel.pack_image_features = pack_image_features


def _log(msg):
    with open(os.environ["STATE_LOG"], "a") as f:
        f.write(msg + "\n")


def pytest_runtest_setup(item):
    global _last, _last_libs
    _last, _last_libs = _state(), _libs()


def pytest_runtest_teardown(item):
    s, libs = _state(), _libs()
    new = sorted(os.path.basename(p) for p in libs - _last_libs)
    tag = "CHANGED" if s != _last else "same"
    _log(f"{tag} {item.nodeid}\n   before: {_last}\n   after:  {s}" + (f"\n   new libs: {new}" if new else ""))
