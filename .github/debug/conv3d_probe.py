# Does a Qwen2-VL-style Conv3d (cuDNN) change the thread locale encoding?
import locale

import torch

print(torch.__version__, "cudnn", torch.backends.cudnn.version(), torch.cuda.get_device_name())


def enc():
    return locale.nl_langinfo(locale.CODESET)


print("start", enc())
for enabled in (True, False):
    torch.backends.cudnn.enabled = enabled
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        conv = torch.nn.Conv3d(3, 32, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False).to("cuda", dtype)
        x = torch.randn(64, 3, 2, 14, 14, device="cuda", dtype=dtype, requires_grad=True)
        conv(x).sum().backward()
        torch.cuda.synchronize()
        print(f"cudnn.enabled={enabled} {dtype}: {enc()}")
