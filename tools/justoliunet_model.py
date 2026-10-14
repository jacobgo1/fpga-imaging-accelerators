#!/usr/bin/env python3
"""numpy model of the justoliunet and justoliunet_bn kernels (needs numpy, not torch).

1D-Justo-LiuNet classifies one pixel from its spectrum: 4 x [Conv1D k=6 valid ->
ReLU -> MaxPool 2], channels 6, 12, 18, 24, then flatten (channel-major, as
torch.flatten) and a dense layer -> 3 scores (0 cloud, 1 land, 2 sea). With 110
bands the spectrum goes 110 -> 105 -> 52 -> 47 -> 23 -> 18 -> 9 -> 4 -> 2, so the
dense layer takes 24 x 2 = 48 values. justoliunet_bn has a BatchNorm after conv1:
conv1 -> bn1 -> ReLU (on the labelled aegean pixels that order gives 96.7 %
accuracy, any later place 63 %), which tools/quantize_weights.py folds into conv1.

  forward_fp32   the trained model from its checkpoint, BatchNorm and all: what
                 PyTorch computes (it reproduces aegean/reference_logits.npy)
  forward        the network the kernels run: the int8 weights (q * 2^-frac,
                 BatchNorm folded) in float, the kernels' reference

The preprocessing is justounetsimple's (keep bands 8-117, z-score with
mu_sd.txt): justounetsimple_model.read_preprocessing / preprocess.

    python tools/justoliunet_model.py      # both models on the aegean pixels: fp32 vs int8
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_weights as ew                      # noqa: E402
from justounetsimple_model import read_preprocessing, preprocess   # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('justoliunet', 'justoliunet_bn')
BN_EPS = 1e-5  # PyTorch BatchNorm default
CONVS = 4


def checkpoint(model):
    found = sorted((ROOT / 'weights/fp32' / model).glob('*.pt'))
    if not found:
        raise FileNotFoundError(f'no checkpoint in weights/fp32/{model}/')
    return found[-1]


def quantized_path(model):
    return ROOT / 'weights/quantized' / model / f'{model}_int8.npz'


def load_fp32(model):
    """The checkpoint's tensors as float64: conv1..4 and fc weight and bias, bn1 if any."""
    _, _, tensors = ew.load(checkpoint(model))
    return {name: value.astype(np.float64) for name, value in tensors.items()}


def load_quantized(model, path=None):
    """{name: q * 2^-frac} for conv1..4 and fc, as float32: the kernel's weights."""
    z = np.load(path or quantized_path(model))
    names = [f'conv{i}.{part}' for i in range(1, CONVS + 1) for part in ('weight', 'bias')] + ['fc.weight', 'fc.bias']
    return {name: z[name].astype(np.float32) * np.float32(2.0 ** -int(z[f'{name}.frac'])) for name in names}


def _conv(h, w, b):
    """Valid 1D convolution: h (N, C_in, L), w (C_out, C_in, K) -> (N, C_out, L - K + 1)."""
    k = w.shape[2]
    length = h.shape[2] - k + 1
    windows = np.stack([h[:, :, i:i + length] for i in range(k)], axis=-1)      # N, C_in, L', K
    return np.einsum('nclk,ock->nol', windows, w) + b[None, :, None]


def _pool(h):
    """MaxPool1d(2): an odd last position is dropped, as in PyTorch."""
    half = h.shape[2] // 2
    return h[:, :, :2 * half].reshape(h.shape[0], h.shape[1], half, 2).max(axis=3)


def _batchnorm(h, w):
    scale = w['bn1.weight'] / np.sqrt(w['bn1.running_var'] + BN_EPS)
    return (h - w['bn1.running_mean'][None, :, None]) * scale[None, :, None] + w['bn1.bias'][None, :, None]


def forward(x, w):
    """Scores (N, 3) of preprocessed spectra x (N, 110). w: conv1..4 and fc weight and bias,
    and bn1 (after conv1, before its ReLU) if w has it."""
    h = np.asarray(x)[:, None, :]
    for i in range(1, CONVS + 1):
        h = _conv(h, w[f'conv{i}.weight'], w[f'conv{i}.bias'])
        if i == 1 and 'bn1.weight' in w:
            h = _batchnorm(h, w)
        h = _pool(np.maximum(h, 0))
    flat = h.reshape(h.shape[0], -1)            # channel-major: (N, C, L) -> c * L + j
    return flat @ w['fc.weight'].T + w['fc.bias']


def forward_fp32(x, model):
    """The trained model, as PyTorch computes it."""
    return forward(np.asarray(x, dtype=np.float64), load_fp32(model))


def aegean_pixels():
    """(raw spectra (N, 120), labels (N,)) of the labelled aegean capture in aegean/."""
    raw = np.loadtxt(ROOT / 'aegean/pixels.txt')
    labels = np.load(ROOT / 'aegean/labels.npy').reshape(-1)
    return raw, labels


def main():
    kept, mean, inv_std = read_preprocessing()
    raw, labels = aegean_pixels()
    x = preprocess(raw, kept, mean, inv_std)
    for model in MODELS:
        fp32 = forward_fp32(x, model)
        line = f'{model:15s} fp32 accuracy {(fp32.argmax(1) == labels).mean():.2%}'
        if quantized_path(model).exists():
            int8 = forward(x, load_quantized(model))
            line += (f', int8 {(int8.argmax(1) == labels).mean():.2%}, same class as fp32 '
                     f'{(int8.argmax(1) == fp32.argmax(1)).mean():.2%}')
        else:
            line += f' (no {quantized_path(model).relative_to(ROOT)}: python tools/quantize_weights.py weights/fp32/{model})'
        print(line + f'  ({len(labels)} aegean pixels)')


if __name__ == '__main__':
    main()
