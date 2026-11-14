"""Export BNCI MAT recordings to the exact NPFEEGNet 2a/2b epoch format."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.io import loadmat

DATASETS = {
    "2a": dict(directory="001-2014", prefix="A", channels=22, classes=4, offset=500),
    "2b": dict(directory="004-2014", prefix="B", channels=3, classes=2, offset=750),
}
BASE_URL = "https://lampx.tugraz.at/~bci/database/"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(path, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as handle:
        temp = Path(handle.name)
    try:
        with urlopen(url, timeout=60) as response, temp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    temp.replace(path)


def export_session(path, dataset):
    setting = DATASETS[dataset]
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = [r for r in np.atleast_1d(content["data"]).ravel() if np.size(r.trial)]
    expected_runs = 6 if dataset == "2a" else (3 if path.stem.endswith("T") else 2)
    if len(runs) != expected_runs:
        raise ValueError(f"{path.name}: expected {expected_runs} task runs")
    epochs, labels, groups = [], [], []
    for group, run in enumerate(runs):
        if float(run.fs) != 250:
            raise ValueError(f"{path.name}: expected 250-Hz recordings")
        signal = np.asarray(run.X)
        starts = np.atleast_1d(run.trial).astype("int64").ravel() - 1
        outcomes = np.atleast_1d(run.y).astype("int64").ravel()
        if len(starts) != len(outcomes) or signal.shape[1] < setting["channels"]:
            raise ValueError(f"{path.name}: invalid signal/event dimensions")
        for onset, label in zip(starts, outcomes, strict=True):
            start = int(onset + setting["offset"])
            stop = start + 1000
            if start < 0 or stop > len(signal):
                raise ValueError(f"{path.name}: epoch exceeds recording boundaries")
            epochs.append(signal[start:stop, :setting["channels"]].T.astype("float32"))
            labels.append(int(label))
            groups.append(group)
    arrays = dict(data=np.stack(epochs), label=np.asarray(labels, dtype="int64")[:, None],
                  group=np.asarray(groups, dtype="int64"))
    if set(labels) != set(range(1, setting["classes"] + 1)) or not np.isfinite(arrays["data"]).all():
        raise ValueError(f"{path.name}: invalid class labels or nonfinite epoch")
    if dataset == "2a" and (len(labels) != 288 or np.bincount(labels)[1:].tolist() != [72] * 4):
        raise ValueError(f"{path.name}: expected 72 trials per class")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mat-dir", type=Path, required=True,
                        help="Directory containing A01T.mat/A01E.mat or B01T.mat/B01E.mat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", choices=range(1, 10), default=list(range(1, 10)))
    parser.add_argument("--download", action="store_true", help="Download missing MAT files from the BNCI host")
    args = parser.parse_args()
    setting = DATASETS[args.dataset]
    args.output.mkdir(parents=True, exist_ok=True)
    names = [f"{setting['prefix']}{s:02d}{part}" for s in args.subjects for part in ("T", "E")]
    targets = [args.output / f"{key}_{suffix}.npy" for key in names for suffix in ("data", "label", "group")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Choose a fresh output directory to preserve existing arrays")
    sources, outputs = {}, {}
    for key in names:
        source = args.mat_dir / f"{key}.mat"
        if not source.exists() and args.download:
            url = f"{BASE_URL}{setting['directory']}/{source.name}"
            print(f"Downloading {url}", flush=True)
            download(source, url)
        arrays = export_session(source, args.dataset)
        sources[source.name] = sha256(source)
        for suffix, array in arrays.items():
            target = args.output / f"{key}_{suffix}.npy"
            np.save(target, array)
            outputs[target.name] = sha256(target)
        print(f"{key}: {arrays['data'].shape}", flush=True)
    manifest = dict(dataset=args.dataset, sample_rate_hz=250, units="microvolts",
                    source_sha256=sources, output_sha256=outputs)
    (args.output / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
