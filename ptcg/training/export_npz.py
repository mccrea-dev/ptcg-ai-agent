"""Export a trained OptionRanker checkpoint to a plain-numpy .npz so the
submission bundle can run inference without torch.

Usage: uv run --group train python -m ptcg.training.export_npz data/train/bc_net.pt -o data/train/bc_net.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arrays = {k: v.numpy() for k, v in ckpt["model"].items()}
    arrays["vocab_json"] = np.frombuffer(
        json.dumps(ckpt["vocab"]).encode(), dtype=np.uint8)
    np.savez_compressed(args.out, **arrays)
    print(f"exported {len(arrays)} arrays -> {args.out}")


if __name__ == "__main__":
    main()
