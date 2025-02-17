from pathlib import Path
import torch
import numpy as np

overwrite = None

for pth_file in Path(__file__).parent.glob("*.pth"):
    npz_file = pth_file.with_suffix(".npz")
    if npz_file.exists():
        if overwrite is None:
            overwrite = input("Overwrite NumPy files? [y/n]")
        if overwrite.lower() != "y":
            continue

    print(f"Converting {pth_file} to {npz_file}")
    state_dict = torch.load(pth_file)
    np.savez_compressed(npz_file, **{
        key: value.cpu().numpy()
        for key, value in state_dict.items()
    })