#!/usr/bin/env python3
"""Export a 1D-Justo-LiuNet checkpoint to the justoliunet HLS kernel (needs numpy).

The kernel takes a raw HYPSO-2 L1a spectrum and does what training did to it
(hypso-onboard-segmentation, dataset_processing/prepare_hypso_dataset.py):

  preprocess: keep 110 of the 120 bands (drop 0-7 and 118-119), then z-score
              each kept band with the training set's mean and std
  network:    4 x [Conv1D k=6 valid -> ReLU -> MaxPool 2], flatten, Dense -> logits
              (1D-Justo-LiuNet, Justo et al., arXiv:2310.16210)

Writes, from one checkpoint and the dataset's mu_sd.txt:

  src/hls/justoliunet/justoliunet_weights.hpp   band selection, mean/std, weights
  tb/justoliunet/justoliunet_vectors.hpp        raw test spectra + expected logits
  tb/data/justoliunet_vectors.txt               the same vectors, for the board

Expected logits come from the numpy forward pass below, not from PyTorch.
A BatchNorm directly after a conv (justoliunet_bn) is folded into that conv.

    python tools/export_justoliunet.py [checkpoint.pt] --mu-sd <dataset>/mu_sd.txt
    python tools/export_justoliunet.py [checkpoint.pt] --placeholder-normalization

The second form uses mean 0 / std 1: the hardware is complete and testable,
but real captures will be classified wrongly until re-exported with --mu-sd.
"""
import argparse
import re
from pathlib import Path

import numpy as np

import pt_reader

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'weights/fp32/justoliunet/justoliunet_pixel1d_e31_miou0-908_K110.pt'
WEIGHTS = 'src/hls/justoliunet/justoliunet_weights.hpp'
RAW_BANDS = 120                                    # HYPSO-2 L1a
DROPPED_BANDS = [0, 1, 2, 3, 4, 5, 6, 7, 118, 119]  # prepare_hypso_dataset.py bands_to_drop
PLACEHOLDER = 'PLACEHOLDER mean 0 / std 1, not the training normalization'
DROPPED_SENTINEL = 1.0e6  # value of dropped bands in test spectra; a kernel reading one fails loudly
BN_EPS = 1e-5  # PyTorch BatchNorm default
MIN_MARGIN = 0.05  # keep test pixels whose top two logits differ at least this much


def load_layers(path):
    sd = pt_reader.state_dict(path)
    convs = []
    for i in range(1, 5):
        w = sd[f'conv{i}.weight'].astype(np.float64)
        b = sd[f'conv{i}.bias'].astype(np.float64)
        if f'bn{i}.weight' in sd:
            scale = sd[f'bn{i}.weight'] / np.sqrt(sd[f'bn{i}.running_var'] + BN_EPS)
            w = w * scale[:, None, None]
            b = (b - sd[f'bn{i}.running_mean']) * scale + sd[f'bn{i}.bias']
        convs.append((w, b))
    fc = (sd['fc.weight'].astype(np.float64), sd['fc.bias'].astype(np.float64))
    return convs, fc


def check_sizes(convs, fc, bands):
    k = convs[0][0].shape[2]
    if convs[0][0].shape[1] != 1 or any(w.shape[2] != k for w, _ in convs):
        raise SystemExit('expected one input channel and the same kernel size in every conv')
    length = bands
    for _ in convs:
        length = (length - k + 1) // 2
    if length * convs[-1][0].shape[0] != fc[0].shape[1]:
        raise SystemExit(f'{bands} bands give {length * convs[-1][0].shape[0]} dense inputs, '
                         f'checkpoint has {fc[0].shape[1]}')
    return k


def read_mu_sd(path):
    """Per-band mean and std from the mu_sd.txt prepare_hypso_dataset.py writes."""
    arrays = re.findall(r'\[([^\]]*)\]', Path(path).read_text())
    if len(arrays) != 2:
        raise SystemExit(f'{path}: expected a mu array and an sd array')
    return [np.array([float(v) for v in a.replace(',', ' ').split()]) for a in arrays]


def preprocessing(mu_sd):
    """Kept band indices, per-band mean and 1/std, all as the kernel stores them."""
    kept = [b for b in range(RAW_BANDS) if b not in DROPPED_BANDS]
    if mu_sd is None:
        mean, std = np.zeros(len(kept)), np.ones(len(kept))
    else:
        mean, std = read_mu_sd(mu_sd)
        if len(mean) != len(kept) or len(std) != len(kept):
            raise SystemExit(f'{mu_sd} has {len(mean)} means and {len(std)} stds; '
                             f'{len(kept)} bands are kept')
    return np.array(kept), mean.astype(np.float32), (1.0 / std).astype(np.float32)


def preprocess(raw, kept, mean, inv_std):
    """The kernel's first stage, in float32 like the hardware: (x - mean) * (1/std)."""
    raw = np.asarray(raw, dtype=np.float32)
    return (raw[:, kept] - mean) * inv_std


def read_preprocessing(header_path=ROOT / WEIGHTS):
    """Band selection and mean/1-std back out of a generated weights header."""
    text = Path(header_path).read_text()

    def array(name):
        body = re.search(rf'{name}\[\w+\] = \{{([^}}]*)\}};', text).group(1)
        return [v.rstrip('f') for v in body.replace(',', ' ').split()]

    kept = np.array([int(v) for v in array('KEPT_BANDS')])
    mean = np.array([float(v) for v in array('BAND_MEAN')], dtype=np.float32)
    inv_std = np.array([float(v) for v in array('BAND_INV_STD')], dtype=np.float32)
    raw_bands = int(re.search(r'constexpr int RAW_BANDS = (\d+);', text).group(1))
    return raw_bands, kept, mean, inv_std, PLACEHOLDER in text


def forward(convs, fc, spectra):
    """Logits for preprocessed spectra of shape (pixels, bands), in float64."""
    h = np.asarray(spectra, dtype=np.float64)[:, None, :]
    for w, b in convs:
        k = w.shape[2]
        length = h.shape[2] - k + 1
        windows = np.stack([h[:, :, j:j + length] for j in range(k)], axis=-1)
        conv = np.einsum('nilk,oik->nol', windows, w) + b[None, :, None]
        conv = np.maximum(conv, 0.0)
        half = length // 2
        h = conv[:, :, :2 * half].reshape(conv.shape[0], conv.shape[1], half, 2).max(axis=3)
    features = h.reshape(h.shape[0], -1)  # PyTorch flatten: channel-major
    return features @ fc[0].T + fc[1]


def synthetic_spectra(rng, count, bands):
    """Smooth random curves in the preprocessed (z-scored) domain, roughly -2..2."""
    x = np.linspace(0.0, 1.0, bands)
    curves = np.cumsum(rng.normal(0.0, 1.0, (count, bands)), axis=1)
    for _ in range(3):
        centre = rng.uniform(0, 1, (count, 1))
        width = rng.uniform(0.03, 0.3, (count, 1))
        height = rng.normal(0, 8, (count, 1))
        curves += height * np.exp(-((x - centre) / width) ** 2)
    lo, hi = curves.min(axis=1, keepdims=True), curves.max(axis=1, keepdims=True)
    return 4.0 * (curves - lo) / (hi - lo) - 2.0


def to_raw(z, kept, mean, inv_std):
    """Raw spectra that preprocess() maps (up to rounding) back to z."""
    raw = np.full((len(z), RAW_BANDS), DROPPED_SENTINEL, dtype=np.float64)
    raw[:, kept] = z / inv_std.astype(np.float64) + mean
    return raw.astype(np.float32)


def pick_vectors(convs, fc, prep, per_class, seed):
    kept, mean, inv_std = prep
    rng = np.random.default_rng(seed)
    pool = synthetic_spectra(rng, 20000, len(kept))
    logits = forward(convs, fc, pool)
    ranked = np.sort(logits, axis=1)
    clear = ranked[:, -1] - ranked[:, -2] >= MIN_MARGIN
    chosen = []
    for cls in range(logits.shape[1]):
        hits = np.flatnonzero(clear & (logits.argmax(axis=1) == cls))[:per_class]
        if len(hits) < per_class:
            print(f'note: only {len(hits)} synthetic pixels land in class {cls}')
        chosen.extend(hits)
    edge = np.stack([np.zeros(len(kept)), np.ones(len(kept)), np.linspace(-2, 2, len(kept))])
    raw = to_raw(np.concatenate([edge, pool[chosen]]), kept, mean, inv_std)
    return raw, forward(convs, fc, preprocess(raw, kept, mean, inv_std))


def c_float(value):
    text = '%.9g' % np.float32(value)
    if not any(ch in text for ch in '.en'):
        text += '.0'
    return text + 'f'


def c_array(values, indent='    '):
    values = np.asarray(values)
    if values.ndim == 1:
        return '{' + ', '.join(c_float(v) for v in values) + '}'
    inner = (',\n' + indent + ' ').join(c_array(v, indent + ' ') for v in values)
    return '{' + inner + '}'


def write_weights(path, source, normalization, convs, fc, k, prep):
    kept, mean, inv_std = prep
    channels = [w.shape[0] for w, _ in convs]
    lines = [
        f'// Generated by tools/export_justoliunet.py from {source}',
        f'// Input normalization: {normalization}',
        '// Do not edit; re-run the exporter instead.',
        '#ifndef JUSTOLIUNET_WEIGHTS_HPP',
        '#define JUSTOLIUNET_WEIGHTS_HPP',
        '',
        'namespace jl {',
        f'constexpr int RAW_BANDS = {RAW_BANDS};',
        f'constexpr int BANDS = {len(kept)};',
        f'constexpr int CLASSES = {fc[0].shape[0]};',
        f'constexpr int K = {k};',
        *[f'constexpr int C{i + 1} = {c};' for i, c in enumerate(channels)],
        f'constexpr bool PLACEHOLDER_NORMALIZATION = {"true" if normalization == PLACEHOLDER else "false"};',
        '',
        '// Preprocessing: network input b = (raw[KEPT_BANDS[b]] - BAND_MEAN[b]) * BAND_INV_STD[b]',
        f'static const int KEPT_BANDS[BANDS] = {{{", ".join(str(b) for b in kept)}}};',
        f'static const float BAND_MEAN[BANDS] = {c_array(mean)};',
        f'static const float BAND_INV_STD[BANDS] = {c_array(inv_std)};',
        '',
    ]
    for i, (w, b) in enumerate(convs, start=1):
        lines.append(f'static const float conv{i}_w[C{i}][{"1" if i == 1 else f"C{i - 1}"}][K] = {c_array(w)};')
        lines.append(f'static const float conv{i}_b[C{i}] = {c_array(b)};')
        lines.append('')
    lines.append(f'static const float fc_w[CLASSES][{fc[0].shape[1]}] = {c_array(fc[0])};')
    lines.append(f'static const float fc_b[CLASSES] = {c_array(fc[1])};')
    lines += ['}  // namespace jl', '', '#endif', '']
    path.write_text('\n'.join(lines))


def write_vectors(header, text, source, inputs, expected):
    header.write_text('\n'.join([
        f'// Generated by tools/export_justoliunet.py from {source}',
        '// Raw spectra and the logits the numpy reference computes for them.',
        '#ifndef JUSTOLIUNET_VECTORS_HPP',
        '#define JUSTOLIUNET_VECTORS_HPP',
        '',
        '#include "justoliunet.hpp"',
        '',
        f'constexpr int JL_VECTORS = {len(inputs)};',
        f'static const float jl_inputs[JL_VECTORS][jl::RAW_BANDS] = {c_array(inputs)};',
        f'static const float jl_expected[JL_VECTORS][jl::CLASSES] = {c_array(expected)};',
        '',
        '#endif',
        '',
    ]))
    rows = [f'# Generated by tools/export_justoliunet.py from {source}',
            f'# One pixel per line: {RAW_BANDS} raw L1a bands | expected logits (numpy reference)']
    for x, y in zip(inputs, expected):
        rows.append(' '.join('%.9g' % v for v in x) + ' | ' + ' '.join('%.9g' % v for v in y))
    text.write_text('\n'.join(rows) + '\n')


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', nargs='?', default=DEFAULT)
    norm = parser.add_mutually_exclusive_group(required=True)
    norm.add_argument('--mu-sd', type=Path, help='mu_sd.txt from the prepared training dataset folder')
    norm.add_argument('--placeholder-normalization', action='store_true',
                      help='mean 0 / std 1 until the real mu_sd.txt is available')
    parser.add_argument('--per-class', type=int, default=4, help='test pixels per predicted class')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--root', type=Path, default=ROOT, help='repository root to write into')
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    source = relative(checkpoint)
    normalization = relative(args.mu_sd) if args.mu_sd else PLACEHOLDER
    convs, fc = load_layers(checkpoint)
    prep = preprocessing(args.mu_sd)
    k = check_sizes(convs, fc, len(prep[0]))
    inputs, expected = pick_vectors(convs, fc, prep, args.per_class, args.seed)

    kernel_dir = args.root / 'src/hls/justoliunet'
    tb_dir = args.root / 'tb/justoliunet'
    data_dir = args.root / 'tb/data'
    for d in (kernel_dir, tb_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_weights(kernel_dir / 'justoliunet_weights.hpp', source, normalization, convs, fc, k, prep)
    write_vectors(tb_dir / 'justoliunet_vectors.hpp', data_dir / 'justoliunet_vectors.txt',
                  source, inputs, expected)
    params = sum(w.size + b.size for w, b in convs) + fc[0].size + fc[1].size
    classes = np.bincount(expected.argmax(axis=1), minlength=fc[0].shape[0])
    print(f'{source}: {RAW_BANDS} raw bands -> {len(prep[0])}, {params} parameters, '
          f'{len(inputs)} test pixels (per class: {classes.tolist()})')
    if args.placeholder_normalization:
        print(f'WARNING: normalization is a {PLACEHOLDER}. Re-export with --mu-sd before '
              'judging results on real captures.')


if __name__ == '__main__':
    main()
