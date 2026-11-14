"""Check processed arrays against all recorded experimental data hashes."""
import argparse
import json
from pathlib import Path

from prepare_data import sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", choices=("2a", "2b"), required=True)
parser.add_argument("--data-dir", type=Path, required=True)
args = parser.parse_args()
reference = Path(__file__).parent / "reference" / f"preprocessing_{args.dataset}.json"
manifest = json.loads(reference.read_text(encoding="utf-8-sig"))
failures = []
for name, expected in manifest["output_sha256"].items():
    path = args.data_dir / name
    if not path.exists() or sha256(path) != expected:
        failures.append(name)
if failures:
    raise SystemExit("Missing or mismatched files: " + ", ".join(failures))
print(f"PASS: all {len(manifest['output_sha256'])} arrays match the recorded {args.dataset} experiments")
