# Academic sources and dependencies

The temporal, depthwise spatial and separable convolution encoder follows
V. J. Lawhern et al., “EEGNet: a compact convolutional neural network for
EEG-based brain–computer interfaces,” Journal of Neural Engineering 15,
056013 (2018). https://doi.org/10.1088/1741-2552/aace8c

The current project-owned NPFEEGNet model files are the same files used by
the November 2026 2a and 2b experiments. Their SHA-256 hashes are recorded in
reference/2a.json and reference/2b.json. The public training entry point
extracts the NPFEEGNet small-learning-rate/S&R workflow from those experiment
runners and exposes data and output locations as command-line arguments.

External runtime dependencies are PyTorch, NumPy and scikit-learn. Their
own licenses apply to those packages.
