"""Run NPFEEGNet inference on prepared microvolt epochs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from engine import SETTINGS
from model import NPFEEGNetRD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SETTINGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    setting = SETTINGS[args.dataset]
    x = np.load(args.input).astype("float32")
    if x.ndim != 3 or x.shape[1:] != (setting["channels"], 1000) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite epochs with shape [trials, channels, 1000]")
    model = NPFEEGNetRD(setting["channels"], setting["classes"], centering="raw").to(args.device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), setting["inference_batch"]):
            batch = torch.from_numpy(x[start:start + setting["inference_batch"], None]).to(args.device)
            scores.append(model(batch).cpu())
    scores = torch.cat(scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, scores=scores.numpy(), probabilities=scores.softmax(1).numpy(),
                        predictions=scores.argmax(1).numpy())


if __name__ == "__main__":
    main()
