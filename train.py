"""Source-only epoch selection, fresh full-source fitting, and target scoring."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from engine import SETTINGS, infer, initialize, train_epoch


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_paths(args, subject, split):
    prefix = SETTINGS[args.dataset]["prefix"]
    return {name: args.data_dir / f"{prefix}{subject:02d}{split}_{name}.npy"
            for name in ("data", "label", "group")}


def load_data(args, subject, split):
    paths = data_paths(args, subject, split)
    x = np.load(paths["data"]).astype("float32")[:, None]
    y = np.load(paths["label"]).ravel().astype("int64") - 1
    groups = np.load(paths["group"]).ravel().astype("int64")
    setting = SETTINGS[args.dataset]
    if x.shape != (len(y), 1, setting["channels"], 1000) or len(groups) != len(y):
        raise ValueError("Unexpected data dimensions")
    if not np.isfinite(x).all() or set(y) != set(range(setting["classes"])):
        raise ValueError("Data must be finite, with original 1-based class labels")
    return x, y, groups


def source_folds(dataset, groups, labels):
    expected = set(range(6)) if dataset == "2a" else {0, 1, 2}
    if set(groups) != expected:
        raise ValueError(f"Expected source groups {sorted(expected)}")
    held_groups = (4, 5) if dataset == "2a" else (2,)
    folds = []
    for held in held_groups:
        train = np.flatnonzero(groups < held)
        valid = np.flatnonzero(groups == held)
        if set(labels[train]) != set(labels) or set(labels[valid]) != set(labels):
            raise ValueError("Each source fold must contain every class")
        folds.append((train, valid))
    return folds


def fit(args, x, y, seed, epochs, folder, validation=None):
    model, optimizer = initialize(seed, args.dataset, args.device)
    tx = torch.from_numpy(x).to(args.device)
    ty = torch.from_numpy(y).to(args.device)
    history = []
    best_key = (1, float("inf"), 0)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, tx, ty, x, y, optimizer, seed, epoch,
                           SETTINGS[args.dataset]["classes"])
        row = dict(epoch=epoch, training_loss=loss)
        if validation is not None:
            vx, vy = validation
            scores, nll, correct = infer(model, vx, vy, args.dataset, args.device)
            row.update(validation_correct=correct, validation_nll=nll)
            key = (-correct, nll, epoch)
            if key < best_key:
                best_key, best_epoch = key, epoch
                torch.save(model.state_dict(), folder / "best.pt")
                np.savez_compressed(folder / "validation.npz", scores=scores.numpy(), labels=vy)
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(folder, epoch, epochs, flush=True)
        save_json(folder / "history.json", history)
    return model, best_epoch


def experiment(args, subject, seed):
    folder = args.output / args.dataset / f"sub{subject:02d}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    x, y, groups = load_data(args, subject, "T")
    folds = source_folds(args.dataset, groups, y)
    code_root = Path(__file__).parent
    code_files = [code_root / name for name in
                  ("train.py", "engine.py", "model/__init__.py",
                   "model/eegnet_encoder.py", "model/npf_eegnet_rd.py")]
    identity = dict(dataset=args.dataset, subject=subject, seed=seed, epochs=args.epochs,
                    source_hashes={k: file_hash(v) for k, v in data_paths(args, subject, "T").items()},
                    code_hashes={str(p.relative_to(code_root)): file_hash(p) for p in code_files})
    if args.stage in ("select", "all"):
        selection_dir = folder / "selection"
        selection_dir.mkdir()
        save_json(folder / "identity.json", identity)
        selected = []
        for i, (train, valid) in enumerate(folds):
            fold_dir = selection_dir / f"fold{i}"
            fold_dir.mkdir()
            _, epoch = fit(args, x[train], y[train], seed, args.epochs, fold_dir,
                           validation=(x[valid], y[valid]))
            selected.append(epoch)
        save_json(folder / "selection.json",
                  dict(fold_epochs=selected, selected_epoch=max(1, int(np.rint(np.median(selected))))))
    if args.stage in ("final", "all"):
        if json.loads((folder / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("Source data, code, or epoch limit changed after selection")
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        reconstructed = []
        for i in range(len(folds)):
            history = json.loads((folder / "selection" / f"fold{i}" / "history.json").read_text())
            if [r["epoch"] for r in history] != list(range(1, args.epochs + 1)):
                raise ValueError("Incomplete source selection history")
            best = min(history, key=lambda r: (-r["validation_correct"], r["validation_nll"], r["epoch"]))
            reconstructed.append(best["epoch"])
        expected = max(1, int(np.rint(np.median(reconstructed))))
        if selection != dict(fold_epochs=reconstructed, selected_epoch=expected):
            raise ValueError("Selected epochs disagree with source histories")
        final_dir = folder / "final"
        final_dir.mkdir()
        model, _ = fit(args, x, y, seed, selection["selected_epoch"], final_dir)
        # Target arrays are first decoded after the complete fixed-duration fit.
        ex, ey, eg = load_data(args, subject, "E")
        scores, nll, correct = infer(model, ex, ey, args.dataset, args.device)
        probability_device = args.device if args.dataset == "2a" else "cpu"
        probabilities = scores.to(probability_device).softmax(1).cpu().numpy()
        predictions = probabilities.argmax(1)
        from sklearn.metrics import cohen_kappa_score, f1_score
        result = dict(dataset=args.dataset, subject=subject, seed=seed,
                      selected_epoch=selection["selected_epoch"], trials=len(ey),
                      accuracy=correct / len(ey), kappa=float(cohen_kappa_score(ey, predictions)),
                      macro_f1=float(f1_score(ey, predictions, average="macro")), nll=nll,
                      target_hashes={k: file_hash(v) for k, v in data_paths(args, subject, "E").items()})
        torch.save(model.state_dict(), final_dir / "model.pt")
        np.savez_compressed(final_dir / "target_predictions.npz", scores=scores.numpy(),
                            labels=ey, groups=eg, probabilities=probabilities, predictions=predictions)
        save_json(final_dir / "result.json", result)
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SETTINGS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)), choices=range(1, 10))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], choices=range(3))
    parser.add_argument("--stage", choices=("select", "final", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=2000, help="Source selection limit, 2000 for reported runs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    stages = ("select", "final") if args.stage == "all" else (args.stage,)
    for stage in stages:
        args.stage = stage
        for subject in args.subjects:
            for seed in args.seeds:
                experiment(args, subject, seed)


if __name__ == "__main__":
    main()
