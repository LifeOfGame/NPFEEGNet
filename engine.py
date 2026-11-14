"""The small-learning-rate/S&R recipe used by the latest 2a and 2b runs."""
import random

import numpy as np
import torch
from torch.nn import functional as F

from model import NPFEEGNetRD

SETTINGS = {
    "2a": dict(prefix="A", channels=22, classes=4, inference_batch=18),
    "2b": dict(prefix="B", channels=3, classes=2, inference_batch=32),
}


def initialize(seed, dataset, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setting = SETTINGS[dataset]
    model = NPFEEGNetRD(setting["channels"], setting["classes"], centering="raw").to(device)
    # The experiment resets dropout RNG after model initialization.
    torch.manual_seed(seed + 1000000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1000000)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002, betas=(.5, .999),
                                 eps=1e-8, weight_decay=0.)
    return model, optimizer


def augment(x, y, count, rng, classes, trace=False):
    """Same-class, channel-synchronous eight-segment recombination."""
    labels = np.arange(count, dtype="int64") % classes
    output = np.empty((count, *x.shape[1:]), dtype="float32")
    donors = np.empty((count, 8), dtype="int64")
    for cls in range(classes):
        rows = np.flatnonzero(labels == cls)
        pool = np.flatnonzero(y == cls)
        if not len(pool):
            raise ValueError("Every class must occur in the source training pool")
        for segment in range(8):
            chosen = rng.choice(pool, size=len(rows), replace=True)
            donors[rows, segment] = chosen
            start, stop = segment * 125, (segment + 1) * 125
            output[rows, ..., start:stop] = x[chosen, ..., start:stop]
    order = rng.permutation(count)
    if trace:
        return output[order], labels[order], donors[order]
    return output[order], labels[order]


def objective(scores, auxiliary, labels):
    return (F.cross_entropy(scores, labels)
            + .5 * F.cross_entropy(auxiliary["raw_logits"], labels)
            + .25 * F.cross_entropy(auxiliary["spectral_logits"], labels))


def train_epoch(model, x, y, pool_x, pool_y, optimizer, seed, epoch, classes):
    model.train()
    rng = np.random.default_rng(seed + 100003 * epoch)
    order = rng.permutation(len(y))
    total, count = 0., 0
    for start in range(0, len(order), 72):
        indices = order[start:start + 72]
        index = torch.as_tensor(indices, device=x.device)
        bx, by = x[index], y[index]
        synthetic, sy = augment(pool_x, pool_y, len(indices), rng, classes)
        bx = torch.cat((bx, torch.from_numpy(synthetic).to(x.device)))
        by = torch.cat((by, torch.from_numpy(sy).to(y.device)))
        optimizer.zero_grad(set_to_none=True)
        for begin in range(0, len(by), 18):
            xx, yy = bx[begin:begin + 18], by[begin:begin + 18]
            scores, auxiliary = model(xx, return_aux=True)
            loss = objective(scores, auxiliary, yy)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training objective")
            (loss * len(yy) / len(by)).backward()
            total += float(loss.detach()) * len(yy)
            count += len(yy)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Missing or nonfinite gradients")
        optimizer.step()
    return total / count


def infer(model, x, y, dataset, device):
    model.eval()
    size = SETTINGS[dataset]["inference_batch"]
    with torch.no_grad():
        chunks = []
        for i in range(0, len(y), size):
            output = model(torch.from_numpy(x[i:i + size].copy()).to(device))
            chunks.append(output if dataset == "2a" else output.cpu())
        scores = torch.cat(chunks)
    if not torch.isfinite(scores).all():
        raise RuntimeError("Nonfinite prediction scores")
    labels = torch.as_tensor(y, dtype=torch.long, device=scores.device)
    return scores.cpu(), float(F.cross_entropy(scores, labels)), int((scores.argmax(1) == labels).sum())
