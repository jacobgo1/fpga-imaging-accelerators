#!/usr/bin/env python3
"""Quantize trained weights to integers for FPGA kernels (needs numpy, not torch).

Post-training quantization of the fp32 checkpoints, in the scheme Vitis AI
(and so the QAT models) uses: symmetric integers with a power-of-two scale
per tensor, value = q * 2^-frac, so rescaling is a shift. Before quantizing,
every BatchNorm is folded into the convolution right before it (all models
here are conv -> BN -> ReLU). For every checkpoint, writes into OUT/<model>/:

  <model>_int8.hpp   each tensor as an int8_t array, with its frac (or scale)
  <model>_int8.npz   the same: q arrays, plus <name>.frac / <name>.scale
  <model>_int8.md    per tensor: scale, worst error, SQNR; which BN went where

and OUT/README.md over all models. Only weights and biases are quantized:
the kernel still has to pick scales for its activations and accumulators.

    python tools/quantize_weights.py                          # all of weights/, int8
    python tools/quantize_weights.py weights/fp32/sp_unet_small --bits 16
    python tools/quantize_weights.py --per-channel            # one frac per output channel
    python tools/quantize_weights.py --scale float            # float scales instead of 2^-frac
"""
import argparse
import json
from pathlib import Path

import numpy as np

import extract_weights as ew

ROOT = ew.ROOT
BN_EPS = 1e-5  # PyTorch BatchNorm default
BN_PARTS = {'weight', 'bias', 'running_mean', 'running_var'}


def groups(tensors):
    """Layer name -> {part: array}, in checkpoint order (conv1 -> {weight, bias})."""
    out = {}
    for name, value in tensors.items():
        layer, _, part = name.rpartition('.')
        out.setdefault(layer, {})[part] = value.astype(np.float64)
    return out


def fold_batchnorm(tensors):
    """Fold every BatchNorm into the conv/linear layer stored right before it.

    Returns (tensors without BN, {conv layer: bn layer}). A BN with no such
    layer in front of it is kept as is.
    """
    layers = groups(tensors)
    folded, pairs, previous = {}, {}, None
    for layer, parts in layers.items():
        is_bn = set(parts) == BN_PARTS and parts['weight'].ndim == 1
        if is_bn and previous and previous[1]['weight'].shape[0] == parts['weight'].shape[0]:
            conv_layer, conv = previous
            scale = parts['weight'] / np.sqrt(parts['running_var'] + BN_EPS)
            conv['weight'] = conv['weight'] * scale.reshape((-1,) + (1,) * (conv['weight'].ndim - 1))
            conv['bias'] = (conv.get('bias', 0.0) - parts['running_mean']) * scale + parts['bias']
            pairs[conv_layer] = layer
            previous = None
            continue
        folded[layer] = parts
        previous = (layer, parts) if 'weight' in parts and parts['weight'].ndim >= 2 else None
    out = {}
    for layer, parts in folded.items():
        for part in ('weight', 'bias', *sorted(set(parts) - {'weight', 'bias'})):
            if part in parts:
                out[f'{layer}.{part}' if layer else part] = parts[part]
    return out, pairs


def tightest_step(x, bits):
    """The smallest step that fits x without clipping, using both ends of the range:
    the most positive value lands on qmax (127) or the most negative on qmin (-128)."""
    qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    return max(float(np.max(x, initial=0.0)) / qmax, float(np.min(x, initial=0.0)) / qmin)


def pow2_frac(x, bits):
    """The power-of-two scale with the smallest squared error (it may clip a few outliers)."""
    qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    step = tightest_step(x, bits)
    if step == 0.0:
        return bits - 1
    no_clip = int(np.floor(-np.log2(step)))  # the finest power-of-two step that is still >= step
    def error(frac):
        q = np.clip(np.round(x * 2.0 ** frac), qmin, qmax)
        return float(np.sum((q * 2.0 ** -frac - x) ** 2))
    return min((no_clip, no_clip + 1, no_clip + 2), key=error)


def quantize(x, bits, scale_kind, per_channel):
    """(q, scales): integers and, per tensor or per output channel, frac or float scale."""
    qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    channels = per_channel and x.ndim >= 2
    rows = x.reshape(x.shape[0], -1) if channels else x.reshape(1, -1)
    if scale_kind == 'pow2':
        fracs = np.array([pow2_frac(r, bits) for r in rows])
        steps = 2.0 ** -fracs.astype(np.float64)
        scales = fracs
    else:
        steps = np.array([tightest_step(r, bits) for r in rows])
        steps = np.where(steps > 0, steps, 1.0)
        scales = steps.astype(np.float32)
    q = np.clip(np.round(rows / steps[:, None]), qmin, qmax).astype(ctype(bits)[1])
    deq = q * steps[:, None]
    return q.reshape(x.shape), (scales if channels else scales[0]), deq.reshape(x.shape)


def ctype(bits):
    for limit, name, dtype in ((8, 'int8_t', np.int8), (16, 'int16_t', np.int16), (32, 'int32_t', np.int32)):
        if bits <= limit:
            return name, dtype
    raise SystemExit('at most 32 bits')


def c_ints(values, indent='    '):
    values = np.asarray(values)
    if values.ndim == 1:
        return '{' + ', '.join(str(int(v)) for v in values) + '}'
    return '{' + (',\n' + indent + ' ').join(c_ints(v, indent + ' ') for v in values) + '}'


def sqnr_db(x, deq):
    noise = float(np.sum((x - deq) ** 2))
    return float('inf') if noise == 0 else 10 * np.log10(float(np.sum(x ** 2)) / noise)


def variant(bits, scale_kind, per_channel):
    return f'int{bits}' + ('_perchannel' if per_channel else '') + ('_floatscale' if scale_kind == 'float' else '')


def scheme_text(bits, scale_kind, per_channel):
    where = 'per output channel' if per_channel else 'per tensor'
    if scale_kind == 'pow2':
        return f'{bits}-bit symmetric, power-of-two scale {where}: value = q * 2^-frac'
    return f'{bits}-bit symmetric, float scale {where}: value = q * scale'


def write_outputs(folder, stem, source, model, meta, results, pairs, bits, scale_kind, per_channel):
    type_name, _ = ctype(bits)
    space = ew.identifier(stem)
    guard = space.upper() + '_HPP'
    folded_by_bn = {f'{conv}.{part}': bn for conv, bn in pairs.items() for part in ('weight', 'bias')}
    lines = [f'// Generated by tools/quantize_weights.py from {source}',
             f'// {ew.describe(model, meta)}',
             f'// {scheme_text(bits, scale_kind, per_channel)}',
             '// BatchNorm folded into the conv before it: ' + (', '.join(f'{b} -> {c}' for c, b in pairs.items())
                                                              or 'none in this model'),
             f'#ifndef {guard}', f'#define {guard}', '', '#include <cstdint>', '', f'namespace {space} {{',
             f'constexpr int BITS = {bits};', '']
    rows = []
    npz = {}
    for name, (x, q, scales, deq) in results.items():
        cname = ew.identifier(name)
        dims = ''.join(f'[{d}]' for d in q.shape)
        error = float(np.max(np.abs(x - deq))) if x.size else 0.0
        sqnr = sqnr_db(x, deq)
        note = f'  (+ {folded_by_bn[name]} folded in)' if name in folded_by_bn else ''
        if scale_kind == 'pow2':
            if np.ndim(scales):
                lines.append(f'// {name}  {list(q.shape)}{note}  frac per output channel  SQNR {sqnr:.1f} dB')
                lines.append(f'static const int {cname}_frac[{len(scales)}] = {c_ints(scales)};')
                scale_text = f'frac {int(np.min(scales))}..{int(np.max(scales))}'
            else:
                frac = int(scales)
                lines.append(f'// {name}  {list(q.shape)}{note}  value = q * 2^-{frac} '
                             f'(fits ap_fixed<{bits}, {bits - frac}>)  SQNR {sqnr:.1f} dB')
                lines.append(f'constexpr int {cname}_frac = {frac};')
                scale_text = f'frac {frac}'
            npz[f'{name}.frac'] = np.asarray(scales)
        else:
            if np.ndim(scales):
                lines.append(f'// {name}  {list(q.shape)}{note}  scale per output channel  SQNR {sqnr:.1f} dB')
                lines.append(f'static const float {cname}_scale[{len(scales)}] = '
                             '{' + ', '.join(ew.c_float(s) for s in scales) + '};')
                scale_text = 'scale per channel'
            else:
                lines.append(f'// {name}  {list(q.shape)}{note}  value = q * scale  SQNR {sqnr:.1f} dB')
                lines.append(f'constexpr float {cname}_scale = {ew.c_float(scales)};')
                scale_text = f'scale {float(scales):.3g}'
            npz[f'{name}.scale'] = np.asarray(scales)
        lines.append(f'static const {type_name} {cname}{dims} = {c_ints(q)};')
        lines.append('')
        npz[name] = q
        rows.append(f'| `{name}` | {list(q.shape)} | {scale_text} | {error:.3g} | {sqnr:.1f} |'
                    + (f' {folded_by_bn[name]} |' if name in folded_by_bn else ' |'))
    lines += [f'}}  // namespace {space}', '', '#endif', '']
    ew.write_unix(folder / f'{stem}.hpp', '\n'.join(lines))
    np.savez(folder / f'{stem}.npz', **npz)
    finite = [sqnr_db(x, deq) for x, _, _, deq in results.values() if np.any(x)]
    worst = min(finite) if finite else float('inf')
    summary = [f'# {model}: {variant(bits, scale_kind, per_channel)}', '',
               f'From `{source}`. {scheme_text(bits, scale_kind, per_channel)}.', '',
               'Folded BatchNorm: ' + (', '.join(f'`{b}` into `{c}`' for c, b in pairs.items()) or 'none') + '.',
               f'Worst tensor SQNR {worst:.1f} dB (signal to quantization noise; '
               'higher is better, int8 usually lands around 30-45 dB).', '',
               '| Tensor | Shape | Scale | Worst error | SQNR (dB) | BN folded in |',
               '| --- | --- | --- | --- | --- | --- |', *rows]
    ew.write_unix(folder / f'{stem}.md', '\n'.join(summary) + '\n')
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='*', help='.pt files or folders (default: weights/)')
    parser.add_argument('--bits', type=int, default=8, help='integer width, 2..32 (default 8)')
    parser.add_argument('--scale', choices=('pow2', 'float'), default='pow2',
                        help='power-of-two scales (default; a shift in hardware) or float scales')
    parser.add_argument('--per-channel', action='store_true', help='one scale per output channel of each weight')
    parser.add_argument('--no-fold-bn', action='store_true', help='quantize BatchNorm tensors separately')
    parser.add_argument('--out', type=Path, default=ROOT / 'weights' / 'quantized')
    args = parser.parse_args()
    if not 2 <= args.bits <= 32:
        raise SystemExit('--bits must be 2..32')

    stem_suffix = variant(args.bits, args.scale, args.per_channel)
    index_file = args.out / 'index.json'
    index = json.loads(index_file.read_text()) if index_file.is_file() else {}
    for path in ew.checkpoints(args.checkpoints):
        model, meta, tensors = ew.load(path)
        source = ew.relative(path)
        if args.no_fold_bn:
            floats, pairs = {k: v.astype(np.float64) for k, v in tensors.items()}, {}
        else:
            floats, pairs = fold_batchnorm(tensors)
        results = {}
        for name, x in floats.items():
            q, scales, deq = quantize(x, args.bits, args.scale, args.per_channel)
            results[name] = (x, q, scales, deq)
        folder = args.out / model
        folder.mkdir(parents=True, exist_ok=True)
        stem = f'{model}_{stem_suffix}'
        worst = write_outputs(folder, stem, source, model, meta, results, pairs, args.bits, args.scale,
                              args.per_channel)
        index[stem] = (f'| [{stem}]({model}/{stem}.md) | {scheme_text(args.bits, args.scale, args.per_channel)} | '
                       f'{len(pairs)} | {worst:.1f} | `{source}` |')
        print(f'{stem:34s} worst SQNR {worst:5.1f} dB, {len(pairs)} BatchNorm folded  -> '
              f'{ew.relative(folder)}/{stem}.hpp')
    index = {stem: row for stem, row in sorted(index.items())
             if (args.out / stem.split('_int')[0] / f'{stem}.hpp').is_file()}
    ew.write_unix(index_file, json.dumps(index, indent=1) + '\n')
    ew.write_unix(args.out / 'README.md', '\n'.join(
        ['# Quantized weights', '', 'Made by `tools/quantize_weights.py`; one folder per model.', '',
         '| Weights | Scheme | BN folded | Worst SQNR (dB) | Checkpoint |', '| --- | --- | --- | --- | --- |',
         *index.values()]) + '\n')
    print(f'Overview: {ew.relative(args.out / "README.md")}')


if __name__ == '__main__':
    main()
