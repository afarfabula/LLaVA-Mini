import importlib
import torch

mods = ["torch", "transformers", "peft", "bitsandbytes", "decord"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "no_version"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__)

print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
