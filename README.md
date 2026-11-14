# NPFEEGNet

NPFEEGNet combines a trial-wise demeaned EEGNet-style raw pathway with a
fixed five-band, response-normalized spectral pathway. The two pathways
have separate encoders. All five bands share the spectral encoder. The
classification scores are combined as raw + α × spectral, with α initialized
to 0.1 and constrained by α = 0.5 sigmoid(a).

This repository contains the latest NPFEEGNet experiments for BCIC-IV-2a and
BCIC-IV-2b with the small-learning-rate + segmentation-and-reconstruction
(S&R) training recipe.

| Dataset | Channels | Classes | Protocol | Accuracy |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 22 | 4 | Official T session → E session, per subject | 82.28% |
| BCIC-IV-2b | 3 | 2 | Sessions 1–3 → sessions 4–5, per subject | 87.50% |

The reported means average seeds 0, 1 and 2 within each subject, then average
the nine subjects equally. Reference run records and selected epoch counts
are in [reference/2a.json](reference/2a.json) and
[reference/2b.json](reference/2b.json).

## Environment

The recorded experiments used Python 3.12.14, PyTorch 2.11.0+cu126,
NumPy 2.4.4 and scikit-learn 1.8.0 on NVIDIA RTX 2080 Ti GPUs.
Install the PyTorch CUDA build suitable for the recorded CUDA 12.6 environment,
then install the remaining pinned requirements.

```bash
python -m pip install -r requirements.txt
```

The model modules are identical to those in the experiment workspaces.
The portable training entry point has been checked against the experiment
training operations. Validation details are recorded in reference/validation.json.
Hardware and runtime differences can affect numerical training trajectories.

## Prepared data

Training starts from the processed NumPy arrays used by the experiments.
Place each dataset's files in its own directory.

```text
data/2a/
  A01T_data.npy    A01T_label.npy    A01T_group.npy
  A01E_data.npy    A01E_label.npy    A01E_group.npy
  ... subjects 02–09
data/2b/
  B01T_data.npy    B01T_label.npy    B01T_group.npy
  B01E_data.npy    B01E_label.npy    B01E_group.npy
  ... subjects 02–09
```

Data arrays have shape [trials, channels, 1000], contain microvolt EEG,
and use 250 Hz sampling. Trial order is preserved. Labels have shape [trials]
or [trials, 1] and use the original 1-based class indices.
Group arrays assign each trial to its original source run or session.

- **2a** uses trial-relative [2, 6) seconds, 288 trials per T/E session,
  22 EEG channels and labels 1–4. Source group IDs are 0–5.
  Selection trains on groups 0–3 and validates on group 4, then trains on
  groups 0–4 and validates on group 5. The final training duration is the
  rounded median of the two selected epochs.
- **2b** uses trial-relative [3, 7) seconds, three EEG channels and labels 1–2.
  Source group IDs are 0–2 for sessions 1–3. Source trials vary by subject.
  Selection trains on groups 0–1 and validates on group 2. The final duration
  is the selected epoch. Target E arrays concatenate sessions 4 and 5.

The model performs raw-pathway temporal demeaning internally.

## Training

```bash
python train.py --dataset 2a --data-dir data/2a --output outputs --device cuda:0
python train.py --dataset 2b --data-dir data/2b --output outputs --device cuda:0
```

By default each command runs all nine subjects and all three seeds sequentially.
Use `--subjects 1` and `--seeds 0` to run one unit. Selection and final
training can also be invoked separately using `--stage select` and
`--stage final`. Final training checks that the source data, code and
selection limit match the saved selection identity.

The fixed recipe uses Adam, learning rate 0.0002, betas (0.5, 0.999),
zero weight decay, 2000 source selection epochs and a constant learning rate.
Each logical batch contains up to 72 original trials plus an equal number
of same-class S&R trials, processed in microbatches of 18 with weighted
gradient accumulation. Each synthetic trial uses eight 125-sample segments
with synchronous donor selection across channels.

Epoch selection maximizes validation correct counts, then minimizes NLL,
then prefers the earlier epoch. Each final model is freshly initialized
and fitted to all source trials for its selected duration. The target
arrays are decoded after that fit. Inference uses chunks of 18 for 2a and
32 for 2b, matching the recorded experiments.

The loss is CE(fused) + 0.5 CE(raw) + 0.25 CE(spectral).
An optional `--epochs 2 --device cpu` run checks the pipeline on CPU.
Use a separate output directory for short checks. The reported recipe uses
2000 selection epochs.

Each completed stage has its own directory. If a run is interrupted, choose
a new output directory or remove its incomplete stage directory before retrying.

## Inference and summaries

```bash
python predict.py --dataset 2a --checkpoint outputs/2a/sub01_seed0/final/model.pt --input data/2a/A01E_data.npy --output outputs/predictions.npz
python summarize.py --dataset 2a --output outputs
python summarize.py --dataset 2b --output outputs
```

Prediction files contain logits, probabilities and zero-based predictions.
Training also stores labels, groups, selected epochs and histories.

See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing and attribution.
