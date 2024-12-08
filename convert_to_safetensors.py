#!/usr/bin/env python3
import sys
import torch
from safetensors.torch import save_file
from pathlib import Path

files = sys.argv[1:]
for f in files:
    f = Path(f)
    if f.suffix in [".pth"]:
        print("Converting", f)
        fn = f.with_suffix(".safetensors")
        if fn.exists():
            print(f"{fn} exists, skipping...")
            continue
        print(f"Loading {f}...")
        try:
            model = torch.load(f, weights_only=True, map_location="cpu")
            weights = {}
            for k in "unet_embedding", "unet_svd", "embeeding":
                subdict = model.pop(k)
                if k == "embeeding":
                    k = "text_embedding"
                for sk in subdict:
                    weights[f"{k}.{sk}"] = subdict[sk]
            weights["alpha"] = model["alpha"]
            weights["beta"] = model["beta"]
            print(f"Saving {fn}...")
            save_file(weights, fn)
            del model
            del weights
        except Exception as ex:
            print(f"ERROR converting {f}: {ex}")

print("Done!")
