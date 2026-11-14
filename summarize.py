"""Average seeds within subject, then average subjects with equal weights."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dataset", choices=("2a", "2b"), required=True)
args = parser.parse_args()
subjects = []
for subject in range(1, 10):
    rows = [json.loads((args.output / args.dataset / f"sub{subject:02d}_seed{seed}" /
                       "final/result.json").read_text()) for seed in range(3)]
    subjects.append(dict(subject=subject, **{
        key: float(np.mean([row[key] for row in rows]))
        for key in ("accuracy", "kappa", "macro_f1", "nll")}))
accuracy = np.array([s["accuracy"] for s in subjects]) * 100
print(json.dumps(dict(dataset=args.dataset, subjects=subjects,
                      accuracy_pct=float(accuracy.mean()),
                      subject_sd_pct=float(accuracy.std(ddof=1))), indent=2))
