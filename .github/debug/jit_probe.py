import locale, torch
def enc(): return locale.nl_langinfo(locale.CODESET)
print("start", enc(), torch.cuda.get_device_name())
x = torch.ones(8, device="cuda"); torch.cuda.synchronize(); print("ones", enc())
y = (x + 1).sum(); torch.cuda.synchronize(); print("add/sum", enc())
p = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda").prod(-1); torch.cuda.synchronize(); print("prod int64", enc())
