#!/usr/bin/env python3
"""numpy model of the spectralpointwise1d kernels (needs numpy, not torch).

1D-SpectralPointwise classifies one pixel from its spectrum with three 1x1 convolutions,
so each layer is a dense layer on the pixel's channels:

    conv 1x1 110 -> 24 -> BatchNorm -> ReLU -> conv 1x1 24 -> 12 -> ReLU -> conv 1x1 12 -> 3

(hypso-onboard-segmentation/model_definition/SpectralPointwise1D.py; every conv has a bias).
tools/quantize_weights.py folds the BatchNorm into the first conv, which leaves three
pointwise layers. 2 964 multiply-adds a pixel, 19x fewer than 1D-Justo-LiuNet.

  forward_fp32   the trained model from its checkpoint, BatchNorm and all: what PyTorch
                 computes
  forward        the network the golden kernel runs: the int8 weights (q * 2^-frac,
                 BatchNorm folded) in float, its reference
  classify_fixed the optimized kernels (spectralpointwise1d_opt and its Zynq-7000 versions)
                 bit for bit, from the raw cube: integers throughout (below)

The preprocessing is justounetsimple's (keep bands 8-117, z-score with mu_sd.txt).

The optimized kernels' integer formats (chosen on the 10 275 labelled aegean pixels):
  input      1D-Justo-LiuNet's: the z-score of each kept band, 12 bits with IN_FRAC = 8
             fraction bits, (dn * A + B) >> PREP_FRAC (tools/justoliunet_model.py,
             src/hls/justoliunet_opt/justoliunet_opt_formats.hpp). The 2D kernels' int8
             with 5 fraction bits costs this model 0.4 points of agreement with fp32 (its
             first layer has the BatchNorm folded in, as justoliunet_bn's).
  activations after convsp1 and convsp2: uint8 (ReLU leaves nothing negative),
             ACT_FRACS fraction bits, the best on aegean.
  scores     int32, OUT_FRAC = 16 fraction bits, then the class, as the other kernels.

    python tools/spectralpointwise1d_model.py          # on the aegean pixels: fp32, int8, fixed
    python tools/spectralpointwise1d_model.py formats  # write src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt_formats.hpp
    python tools/spectralpointwise1d_model.py reference IMAGE   # IMAGE/reference_spectralpointwise1d.npy:
        # what every optimized kernel (ZCU104, Zynq-7030, Zynq-7020) writes for IMAGE/cube.npy,
        # bit for bit; the OPU's classify takes it as its reference
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_weights as ew                      # noqa: E402
from justounetsimple_model import read_preprocessing, preprocess   # noqa: E402,F401
from justoliunet_model import (IN_FRAC, IN_MAX, OUT_FRAC, ACT_MAX, preprocess_fixed,   # noqa: E402,F401
                               aegean_pixels, _round_shift)

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'spectralpointwise1d'
LAYERS = ('convsp1', 'convsp2', 'convsp3')
BN_EPS = 1e-5  # PyTorch BatchNorm default

ACT_FRACS = (4, 4)        # fraction bits after convsp1 and convsp2 (see the docstring)
FORMATS_HEADER = ROOT / 'src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt_formats.hpp'


def checkpoint():
    found = sorted((ROOT / 'weights/fp32' / MODEL).glob('*.pt'))
    if not found:
        raise FileNotFoundError(f'no checkpoint in weights/fp32/{MODEL}/')
    return found[-1]


def quantized_path():
    return ROOT / 'weights/quantized' / MODEL / f'{MODEL}_int8.npz'


def load_fp32():
    """The checkpoint's tensors as float64, the convs' weights as (OUT, IN) matrices."""
    _, _, tensors = ew.load(checkpoint())
    return {name: (value.reshape(value.shape[0], -1) if value.ndim == 4 else value).astype(np.float64)
            for name, value in tensors.items()}


def load_quantized(path=None):
    """{name: q * 2^-frac} for the three convs' weight (OUT, IN) and bias, as float32."""
    z = np.load(path or quantized_path())
    out = {}
    for layer in LAYERS:
        for part in ('weight', 'bias'):
            name = f'{layer}.{part}'
            v = z[name].astype(np.float32) * np.float32(2.0 ** -int(z[f'{name}.frac']))
            out[name] = v.reshape(v.shape[0], -1) if part == 'weight' else v
    return out


def forward(x, w):
    """Scores (N, 3) of preprocessed spectra x (N, 110). w: the three convs' weight (OUT, IN)
    and bias, and bnsp1 (after convsp1, before its ReLU) if w has it."""
    h = np.asarray(x) @ w['convsp1.weight'].T + w['convsp1.bias']
    if 'bnsp1.weight' in w:
        scale = w['bnsp1.weight'] / np.sqrt(w['bnsp1.running_var'] + BN_EPS)
        h = (h - w['bnsp1.running_mean']) * scale + w['bnsp1.bias']
    h = np.maximum(h, 0)
    h = np.maximum(h @ w['convsp2.weight'].T + w['convsp2.bias'], 0)
    return h @ w['convsp3.weight'].T + w['convsp3.bias']


def forward_fp32(x):
    """The trained model, as PyTorch computes it."""
    return forward(np.asarray(x, dtype=np.float64), load_fp32())


def load_quantized_int(path=None):
    """{name: (q as int64, frac)} for the three convs' weight (OUT, IN) and bias."""
    z = np.load(path or quantized_path())
    out = {}
    for name in z.files:
        if not name.endswith('.frac'):
            v = z[name].astype(np.int64)
            out[name] = (v.reshape(v.shape[0], -1) if v.ndim == 4 else v, int(z[f'{name}.frac']))
    return out


def classify_fixed(raw, q=None, prep=None, fracs=ACT_FRACS, in_frac=IN_FRAC, in_max=IN_MAX):
    """What spectralpointwise1d_opt writes for raw pixels (N, 120): (N, 4) int32, the 3
    scores * 2^OUT_FRAC and the class (the first highest score), bit for bit. Each layer:
    sum(act * q) + bias * 2^(sums' frac - bias' frac), in integers; after the first two,
    ReLU, round off down to the layer's ACT_FRAC, saturate at 255. (fracs, in_frac and
    in_max are only for trying other formats.)"""
    q = q or load_quantized_int()
    if in_frac == IN_FRAC and in_max == IN_MAX:
        h = preprocess_fixed(np.asarray(raw).reshape(-1, 120), prep)
    else:   # another input format: the float z-score, rounded (only to compare formats)
        kept, mean, inv_std = read_preprocessing()
        z = preprocess(np.asarray(raw).reshape(-1, 120), kept, mean, inv_std).astype(np.float64)
        h = np.clip(np.rint(z * 2.0 ** in_frac), -in_max, in_max).astype(np.int64)
    fa = in_frac
    for i, layer in enumerate(LAYERS[:2]):
        (w, fw), (b, fb) = q[f'{layer}.weight'], q[f'{layer}.bias']
        assert fa + fw >= fb and fa + fw > fracs[i], 'the shifts must be as the kernel has them'
        s = h @ w.T + (b << (fa + fw - fb))
        h = np.clip(_round_shift(s, fa + fw - fracs[i]), 0, ACT_MAX)
        fa = fracs[i]
    (w, fw), (b, fb) = q['convsp3.weight'], q['convsp3.bias']
    assert fa + fw >= fb and fa + fw <= OUT_FRAC
    scores = (h @ w.T + (b << (fa + fw - fb))) << (OUT_FRAC - fa - fw)
    return np.concatenate([scores, scores.argmax(1)[:, None]], axis=1).astype(np.int32)


def reference(image, chunk=65536):
    """IMAGE/reference_spectralpointwise1d.npy, (H, W, 4) int32: classify_fixed on
    IMAGE/cube.npy, chunk pixels at a time."""
    image = Path(image)
    cube = np.load(image / 'cube.npy', mmap_mode='r')
    h, w, bands = cube.shape
    if bands != 120:
        raise ValueError(f'{image}/cube.npy has {bands} bands; the kernels take the raw 120')
    pixels = np.asarray(cube).reshape(h * w, bands)
    q = load_quantized_int()
    out = np.empty((h * w, 4), dtype=np.int32)
    for start in range(0, h * w, chunk):
        out[start:start + chunk] = classify_fixed(pixels[start:start + chunk], q)
    path = image / f'reference_{MODEL}.npy'
    np.save(path, out.reshape(h, w, 4))
    return path


def formats_header():
    """src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt_formats.hpp: the activations'
    formats above, for the optimized kernels (the input's are justoliunet_opt_formats.hpp's)."""
    lines = ['// Generated by tools/spectralpointwise1d_model.py -- do not edit; run',
             '//     python tools/spectralpointwise1d_model.py formats',
             '// after changing the formats there.',
             '//',
             "// The optimized spectralpointwise1d kernels' activation formats (tools/spectralpointwise1d_model.py",
             '// has why): after convsp1 and convsp2, uint8 with ACT_FRAC<i> fraction bits. The input',
             "// (12-bit z-score) and the scores are 1D-Justo-LiuNet's: justoliunet_opt_formats.hpp.",
             '#ifndef SPECTRALPOINTWISE1D_OPT_FORMATS_HPP', '#define SPECTRALPOINTWISE1D_OPT_FORMATS_HPP', '',
             f'namespace {MODEL}_formats {{',
             *[f'constexpr int ACT_FRAC{i + 1} = {f};' for i, f in enumerate(ACT_FRACS)],
             f'}}  // namespace {MODEL}_formats', '', '#endif', '']
    return '\n'.join(lines)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'reference':
        print(f'wrote {reference(sys.argv[2])}')
        return
    if sys.argv[1:] == ['formats']:
        FORMATS_HEADER.parent.mkdir(parents=True, exist_ok=True)
        FORMATS_HEADER.write_text(formats_header())
        print(f'wrote {FORMATS_HEADER.relative_to(ROOT)}')
        return
    kept, mean, inv_std = read_preprocessing()
    raw, labels = aegean_pixels()
    x = preprocess(raw, kept, mean, inv_std)
    fp32 = forward_fp32(x)
    line = f'{MODEL} fp32 accuracy {(fp32.argmax(1) == labels).mean():.2%}'
    if quantized_path().exists():
        int8 = forward(x, load_quantized())
        fixed = classify_fixed(raw)[:, 3]
        line += (f', int8 {(int8.argmax(1) == labels).mean():.2%} (same class as fp32 '
                 f'{(int8.argmax(1) == fp32.argmax(1)).mean():.2%}), fixed {(fixed == labels).mean():.2%} '
                 f'(same class as fp32 {(fixed == fp32.argmax(1)).mean():.2%})')
    else:
        line += f' (no {quantized_path().relative_to(ROOT)}: python tools/quantize_weights.py weights/fp32/{MODEL})'
    print(line + f'  ({len(labels)} aegean pixels)')


if __name__ == '__main__':
    main()
