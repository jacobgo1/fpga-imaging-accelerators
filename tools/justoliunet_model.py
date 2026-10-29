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
  forward        the network the golden kernels run: the int8 weights (q * 2^-frac,
                 BatchNorm folded) in float, their reference
  classify_fixed the optimized kernels (justoliunet_opt, justoliunet_bn_opt) bit for
                 bit, from the raw cube: integers throughout (below)

The preprocessing is every model's (keep bands 8-117, z-score with mu_sd.txt):
tools/preprocessing.py.

The optimized kernels' integer formats (chosen on the 10 275 labelled aegean pixels):
  input      the z-score of each kept band, 12 bits with IN_FRAC = 8 fraction bits
             (+-8 sigma), (dn * A + B) >> PREP_FRAC. The 2D kernel's int8 with 5 bits
             costs justoliunet_bn 1.2 points of agreement with fp32: its folded conv1
             weights are up to 16x larger, and 1 % of z-scores pass its +-3.97.
  activations after each conv block: uint8 (ReLU leaves nothing negative), ACT_FRACS
             fraction bits per block and model; as good as int16 here, and two of
             their products fit one DSP.
  scores     int32, OUT_FRAC = 16 fraction bits, then the class, as justounetsimple_opt.

    python tools/justoliunet_model.py          # both models on the aegean pixels: fp32, int8, fixed
    python tools/justoliunet_model.py formats  # write src/hls/justoliunet_opt/justoliunet_opt_formats.hpp
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_weights as ew                      # noqa: E402
import preprocessing                              # noqa: E402
from preprocessing import read_preprocessing, preprocess, MU_SD   # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('justoliunet', 'justoliunet_bn')
BN_EPS = 1e-5  # PyTorch BatchNorm default
CONVS = 4

# The optimized kernels' formats (see the docstring).
IN_FRAC, IN_MAX, PREP_FRAC = 8, 2047, 19       # input: 12-bit z-score, 8 fraction bits
ACT_MAX = 255                                    # activations: uint8
ACT_FRACS = {'justoliunet': (4, 4, 2, 3),        # fraction bits after blocks 1-4: one more
             'justoliunet_bn': (4, 4, 2, 2)}     # than the largest value needs, the best on aegean
OUT_FRAC = 16                                    # scores: score * 2^16
FORMATS_HEADER = ROOT / 'src/hls/justoliunet_opt/justoliunet_opt_formats.hpp'


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


def prep_constants(mu_sd=MU_SD):
    """The z-score as integers in this 12-bit format: preprocessing.prep_constants."""
    return preprocessing.prep_constants(mu_sd, IN_FRAC, PREP_FRAC)


def preprocess_fixed(raw, prep=None):
    """Raw L1a values (..., 120 integers) -> the optimized kernels' input (..., 110)."""
    return preprocessing.preprocess_fixed(raw, prep if prep is not None else prep_constants(),
                                          PREP_FRAC, IN_MAX, np.int64)


def load_quantized_int(model, path=None):
    """{name: (q as int64, frac)} for conv1..4 and fc weight and bias."""
    z = np.load(path or quantized_path(model))
    return {name: (z[name].astype(np.int64), int(z[f'{name}.frac'])) for name in z.files if not name.endswith('.frac')}


def _round_shift(s, shift):
    return s if shift == 0 else (s + (1 << (shift - 1))) >> shift


def classify_fixed(raw, model, q=None, prep=None):
    """What justoliunet_opt (or justoliunet_bn_opt) writes for raw pixels (N, 120): (N, 4)
    int32, the 3 scores * 2^OUT_FRAC and the class (the first highest score), bit for bit.
    Each conv: sum(act * q) + bias * 2^(sums' frac - bias' frac), in integers; then ReLU,
    round off down to the block's ACT_FRAC, saturate at 255; then max-pool."""
    q = q or load_quantized_int(model)
    fracs = ACT_FRACS[model]
    h = preprocess_fixed(np.asarray(raw).reshape(-1, 120), prep)[:, None, :]      # N, C, L
    fa = IN_FRAC
    for i in range(1, CONVS + 1):
        (w, fw), (b, fb) = q[f'conv{i}.weight'], q[f'conv{i}.bias']
        assert fa + fw >= fb and fa + fw > fracs[i - 1], 'the shifts must be as the kernel has them'
        k = w.shape[2]
        length = h.shape[2] - k + 1
        windows = np.stack([h[:, :, j:j + length] for j in range(k)], axis=-1)
        s = np.einsum('nclk,ock->nol', windows, w) + (b << (fa + fw - fb))[None, :, None]
        h = _pool(np.clip(_round_shift(s, fa + fw - fracs[i - 1]), 0, ACT_MAX))
        fa = fracs[i - 1]
    (w, fw), (b, fb) = q['fc.weight'], q['fc.bias']
    assert fa + fw >= fb and fa + fw <= OUT_FRAC
    s = h.reshape(len(h), -1) @ w.T + (b << (fa + fw - fb))       # channel-major flatten
    scores = s << (OUT_FRAC - fa - fw)
    return np.concatenate([scores, scores.argmax(1)[:, None]], axis=1).astype(np.int32)


def formats_header():
    """src/hls/justoliunet_opt/justoliunet_opt_formats.hpp: the formats above, for the
    optimized kernels."""
    a, b = prep_constants()
    def ints(v):
        return ',\n    '.join(', '.join(str(int(x)) for x in v[i:i + 8]) for i in range(0, len(v), 8))
    lines = ['// Generated by tools/justoliunet_model.py from mu_sd.txt -- do not edit; run',
             '//     python tools/justoliunet_model.py formats',
             '// after changing mu_sd.txt or the formats there.',
             '//',
             "// The optimized 1D kernels' integer formats (tools/justoliunet_model.py has why).",
             '// Input: kept band k is raw band 8 + k; 110 and 111 are raw 118 and 119, zeroed by',
             '// A = B = 0:',
             '//     x * 2^IN_FRAC = round((dn - mean) / std * 2^IN_FRAC) = (dn * A[k] + B[k]) >> PREP_FRAC',
             '// (to within one unit), saturated to +-IN_MAX (12 bits).',
             '// Activations after conv block i: uint8 with ACT_FRAC<i> fraction bits, per model.',
             '#ifndef JUSTOLIUNET_OPT_FORMATS_HPP', '#define JUSTOLIUNET_OPT_FORMATS_HPP', '',
             '#include <stdint.h>', '', 'namespace justoliunet_opt_formats {',
             f'constexpr int IN_FRAC = {IN_FRAC};', f'constexpr int IN_MAX = {IN_MAX};',
             f'constexpr int PREP_FRAC = {PREP_FRAC};', 'constexpr int FIRST_BAND = 8;',
             f'constexpr int ACT_MAX = {ACT_MAX};', f'constexpr int OUT_FRAC = {OUT_FRAC};', '',
             f'static const int32_t A[112] = {{\n    {ints(a)}}};', '',
             f'static const int64_t B[112] = {{\n    {ints(b)}}};', '', '}  // namespace justoliunet_opt_formats', '']
    for model, fr in ACT_FRACS.items():
        lines += [f'namespace {model}_formats {{',
                  *[f'constexpr int ACT_FRAC{i + 1} = {f};' for i, f in enumerate(fr)],
                  f'}}  // namespace {model}_formats', '']
    lines += ['#endif', '']
    return '\n'.join(lines)


def aegean_pixels():
    """(raw spectra (N, 120), labels (N,)) of the labelled aegean capture in aegean/."""
    raw = np.loadtxt(ROOT / 'aegean/pixels.txt')
    labels = np.load(ROOT / 'aegean/labels.npy').reshape(-1)
    return raw, labels


def main():
    if sys.argv[1:] == ['formats']:
        FORMATS_HEADER.parent.mkdir(parents=True, exist_ok=True)
        FORMATS_HEADER.write_text(formats_header())
        print(f'wrote {FORMATS_HEADER.relative_to(ROOT)}')
        return
    kept, mean, inv_std = read_preprocessing()
    raw, labels = aegean_pixels()
    x = preprocess(raw, kept, mean, inv_std)
    for model in MODELS:
        fp32 = forward_fp32(x, model)
        line = f'{model:15s} fp32 accuracy {(fp32.argmax(1) == labels).mean():.2%}'
        if quantized_path(model).exists():
            int8 = forward(x, load_quantized(model))
            fixed = classify_fixed(raw, model)[:, 3]
            line += (f', int8 {(int8.argmax(1) == labels).mean():.2%} (same class as fp32 '
                     f'{(int8.argmax(1) == fp32.argmax(1)).mean():.2%}), fixed {(fixed == labels).mean():.2%} '
                     f'(same class as fp32 {(fixed == fp32.argmax(1)).mean():.2%})')
        else:
            line += f' (no {quantized_path(model).relative_to(ROOT)}: python tools/quantize_weights.py weights/fp32/{model})'
        print(line + f'  ({len(labels)} aegean pixels)')


if __name__ == '__main__':
    main()
