"""SpecPW-UNet with 8-bit activations: how much accuracy would they cost?

Today every activation is int16 with ACT_FRAC = 10 fraction bits (specpw_unet_model.py).
Two multiplies per DSP (dsp_pack.hpp) need 8-bit activations, so this runs the same
bit-exact integer network with each layer's activations ReLU'd, rounded and saturated to
8 bits instead:

    int8   0..127  (act8_t, what conv3x3_2mul / pointwise_packed take: groups of 8 on DSP48E2)
    uint8  0..255  (uact_t, as the 1D networks: groups of 4 on DSP48E2)

Each layer gets its own power-of-two scale F (activation = value * 2^F): the largest F for
which the layer's largest float activation (int8 weights) on a calibration capture fits in
127 or 255, or its 99.99th percentile. up2.1 and the skip (double1.block.3) share one F,
since concat joins them. The weights, the input z-scores (IN_FRAC) and the scores
(OUT_FRAC) are as today.

The captures are cut into 32 x 32 patches as the kernels do (last row and column repeated),
and every conv sums exactly (in float64, all sums are far below 2^53). With int16 the
script reproduces what the boards measured (97.48 / 97.32 / 97.55 % on aegantsea1).

    python tools/specpw_unet_act8.py [EVAL_CAPTURE [CALIBRATION_CAPTURE]]
    python tools/specpw_unet_act8.py --write [CALIBRATION_CAPTURE]
    python tools/specpw_unet_act8.py --check [CAPTURE ...]

--write fixes the uint8 kernels' scales (sp_unet_SIZE_opt_u8): uint8, p100 on the calibration
capture, written to src/models/SpecPW-UNet/specpw_unet_act8_fracs.hpp, which the kernels,
their testbenches and specpw_unet_model.act8_fracs read. The scales are constants, the same
for every image; a new image whose activations go past a layer's range saturates at 255.
--check runs those fixed scales on each capture (default: every NAME-l1a.nc/ in the
repository) against int16: accuracy, classes kept and how many activations saturated.

Captures are the repository's NAME-l1a.nc/ folders (the .nc and its _labels.npy); the
defaults are aegantsea1 (evaluated) and saldanha (calibration, also evaluated). Reading the
.nc needs h5py (pip install h5py). Results: docs/experiments/specpw_unet_8bit_activations.md.
"""
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing  # noqa: E402
import specpw_unet_model as m  # noqa: E402

EVAL = 'aegantsea1_2026-01-20T09-00-18Z'
CALIBRATION = 'saldanha_2026-01-18T09-13-20Z'
REMAP = {1: 0, 2: 1, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}   # as tools/image.py; others unlabelled
P = m.PATCH


def load(name):
    """(raw cube H x W x 120, labels H x W with -1 for unlabelled) of a capture folder."""
    try:
        import h5py
    except ImportError:
        raise SystemExit('reading the .nc captures needs h5py: pip install h5py')
    folder = ROOT / f'{name}-l1a.nc'
    with h5py.File(folder / f'{name}-l1a.nc', 'r') as f:
        cube = f['products/dn'][()]
    raw = np.load(folder / f'{name}-l1a_labels.npy')
    labels = np.full(raw.shape, -1)
    for k, v in REMAP.items():
        labels[raw == k] = v
    return cube, labels


def to_patches(img):
    """H x W x C -> (patches N x 32 x 32 x C in raster order, shape to undo it)."""
    h, w = img.shape[:2]
    hp, wp = -(-h // P) * P, -(-w // P) * P
    img = np.pad(img, ((0, hp - h), (0, wp - w), (0, 0)), mode='edge')
    patches = img.reshape(hp // P, P, wp // P, P, -1).transpose(0, 2, 1, 3, 4).reshape(-1, P, P, img.shape[-1])
    return patches, (h, w, hp, wp)


def from_patches(patches, shape):
    h, w, hp, wp = shape
    return patches.reshape(hp // P, wp // P, P, P, -1).transpose(0, 2, 1, 3, 4).reshape(hp, wp, -1)[:h, :w]


def conv(x, w):
    """Same-padded conv of every patch: x (N, h, w, in), w (out, in, k, k), in float64."""
    k = w.shape[2]
    pad = (k - 1) // 2
    h, wd = x.shape[1:3]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out = np.zeros(x.shape[:3] + (w.shape[0],))
    for i in range(k):
        for j in range(k):
            out += xp[:, i:i + h, j:j + wd] @ w[:, :, i, j].T.astype(np.float64)
    return out


def pool(x):
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def up(x):
    return x.repeat(2, 1).repeat(2, 2)


class Act:
    """A map and its fraction bits."""
    def __init__(self, v, frac):
        self.v, self.frac = v, frac


def network(x, conv_relu, final):
    """specpw_unet_model._network on batches of patches, carrying each map's scale."""
    for layer in m.LAYERS[:3]:
        x = conv_relu(x, layer)
    skip = conv_relu(conv_relu(x, 'double1.block.0'), 'double1.block.3')
    x = conv_relu(conv_relu(Act(pool(skip.v), skip.frac), 'double2.block.0'), 'double2.block.3')
    x = conv_relu(Act(up(pool(x.v)), x.frac), 'up1.1')
    x = conv_relu(conv_relu(x, 'double3.block.0'), 'double3.block.3')
    x = conv_relu(Act(up(x.v), x.frac), 'up2.1')
    assert x.frac == skip.frac, 'concat needs one scale'
    x = Act(np.concatenate([x.v, skip.v], axis=-1), x.frac)
    x = conv_relu(conv_relu(x, 'double4.block.0'), 'double4.block.3')
    return final(x)


def round_shift(s, shift):
    """Round half up, as the kernels (exact on integer-valued float64)."""
    return np.floor(s * 2.0 ** -shift + 0.5) if shift > 0 else s * 2.0 ** -shift


def forward_fixed(xq, qlayers, fracs, act_max, stats=None):
    """Integer network: layer l's activations ReLU'd, rounded to fracs[l] fraction bits and
    saturated at act_max. -> scores * 2^OUT_FRAC."""
    def conv_relu(x, layer):
        w, wf, b, bf = qlayers[layer]
        sum_frac = x.frac + wf
        s = conv(x.v, w) + b.astype(np.float64) * 2.0 ** (sum_frac - bf)
        y = np.maximum(round_shift(s, sum_frac - fracs[layer]), 0)
        if stats is not None:
            stats['saturated'] += int((y > act_max).sum())
            stats['values'] += y.size
        return Act(np.minimum(y, act_max), fracs[layer])

    def final(x):
        w, wf, b, bf = qlayers['final']
        sum_frac = x.frac + wf
        return round_shift(conv(x.v, w) + b * 2.0 ** (sum_frac - bf), sum_frac - m.OUT_FRAC)

    return network(Act(xq.astype(np.float64), m.IN_FRAC), conv_relu, final)


def float_activation_size(xf, layers, percentile):
    """Per layer: the given percentile of its non-zero float activations (int8 weights)."""
    size = {}

    def conv_relu(x, layer):
        w, b = layers[layer]
        y = np.maximum(conv(x.v, w) + b, 0)
        nonzero = y[y > 0]
        size[layer] = float(np.percentile(nonzero, percentile)) if nonzero.size else 1.0
        return Act(y, 0)

    network(Act(xf.astype(np.float64), 0), conv_relu, lambda x: None)
    return size


def scales(size, act_max):
    """Per layer: the largest F with size * 2^F <= act_max; up2.1 and the skip share theirs."""
    fracs = {layer: int(np.floor(np.log2(act_max / v))) for layer, v in size.items()}
    shared = min(fracs['up2.1'], fracs['double1.block.3'])
    fracs['up2.1'] = fracs['double1.block.3'] = shared
    return fracs


CALIBRATION_PERCENTILE, CALIBRATION_MAX = 100.0, 255   # the uint8 kernels': p100, 0..255


def calibrate(xf, qlayers):
    """The uint8 kernels' fraction bits {layer: F} from float calibration patches xf."""
    sizes = float_activation_size(xf, m.dequantize(qlayers), CALIBRATION_PERCENTILE)
    return scales(sizes, CALIBRATION_MAX), sizes


def header(fracs_by_size, sizes_by_size, calibration):
    """specpw_unet_act8_fracs.hpp: a namespace of fraction bits per size."""
    lines = ['#pragma once', '',
             '// Generated by python tools/specpw_unet_act8.py --write: do not edit.',
             "// The uint8 activations of sp_unet_SIZE_opt_u8: layer L's activation a means",
             '// a * 2^-L, 0 .. 255, the same for every image. L is the largest that fits the',
             f"// layer's largest float activation (int8 weights, p{CALIBRATION_PERCENTILE:g}) on {calibration}",
             '// in 255; up2_1 and the skip (double1_block_3) share theirs, for the concat.']
    for size, fracs in fracs_by_size.items():
        lines += ['', f'namespace specpw_unet_act8_{size} {{']
        for layer in m.LAYERS:
            if layer == 'final':
                continue
            lines.append(f'constexpr int {layer.replace(".", "_")} = {fracs[layer]};'
                         f'   // largest {sizes_by_size[size][layer]:.3f}, up to {255 * 2.0 ** -fracs[layer]:g}')
        lines.append('}')
    return '\n'.join(lines) + '\n'


def write(cal_name):
    """Calibrates every size on cal_name and writes m.ACT8_FRACS."""
    prep = preprocessing.read_preprocessing()
    cube, _ = load(cal_name)
    xf, _ = to_patches(preprocessing.preprocess(cube.astype(np.float32), *prep))
    fracs, sizes = {}, {}
    for size_name in m.SIZES:
        fracs[size_name], sizes[size_name] = calibrate(xf, m.load_quantized(size_name))
    m.ACT8_FRACS.write_text(header(fracs, sizes, cal_name.split('_')[0]), newline='\n')
    print(f'wrote {m.ACT8_FRACS.relative_to(ROOT)}')


def check(names):
    """The uint8 kernels' fixed scales (m.act8_fracs) against int16 on each capture."""
    names = names or sorted(f.name[:-len('-l1a.nc')] for f in ROOT.glob('*-l1a.nc') if f.is_dir())
    prep = preprocessing.prep_constants()
    print('| Capture | Size | int16 | uint8 | difference | same class as int16 | saturated |')
    print('|---|---|---|---|---|---|---|')
    for name in names:
        cube, labels = load(name)
        xq, shape = to_patches(preprocessing.preprocess_fixed(cube, prep))
        for size_name in m.SIZES:
            qlayers = m.load_quantized(size_name)
            acc16, classes16 = accuracy(forward_fixed(xq, qlayers, m.int16_fracs(), m.ACT_MAX), shape, labels)
            stats = {'saturated': 0, 'values': 0}
            acc8, classes8 = accuracy(forward_fixed(xq, qlayers, m.act8_fracs(size_name), m.U8_MAX, stats),
                                      shape, labels)
            print(f'| {name.split("_")[0]} | {size_name} | {acc16:.2f} % | {acc8:.2f} % | {acc8 - acc16:+.2f} pp | '
                  f'{100 * (classes8 == classes16).mean():.2f} % | '
                  f'{100 * stats["saturated"] / stats["values"]:.4f} % |', flush=True)


def accuracy(scores, shape, labels):
    classes = from_patches(scores, shape).argmax(-1)
    labelled = labels >= 0
    return 100 * (classes[labelled] == labels[labelled]).mean(), classes


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--write':
        write(sys.argv[2] if len(sys.argv) > 2 else CALIBRATION)
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check(sys.argv[2:])
        return
    eval_name = sys.argv[1] if len(sys.argv) > 1 else EVAL
    cal_name = sys.argv[2] if len(sys.argv) > 2 else CALIBRATION
    prep = preprocessing.prep_constants()
    kept, mean, inv_std = preprocessing.read_preprocessing()
    data = {}
    for name in dict.fromkeys((eval_name, cal_name)):
        cube, labels = load(name)
        xq, shape = to_patches(preprocessing.preprocess_fixed(cube, prep))
        xf, _ = to_patches(preprocessing.preprocess(cube.astype(np.float32), kept, mean, inv_std))
        data[name] = (xq, xf, shape, labels)

    for size_name in m.SIZES:
        start = time.time()
        qlayers = m.load_quantized(size_name)
        layers = m.dequantize(qlayers)
        print(f'sp_unet_{size_name}')
        for name, (xq, _, shape, labels) in data.items():
            short = name.split('_')[0]
            today = {layer: m.ACT_FRAC for layer in m.LAYERS}
            acc16, classes16 = accuracy(forward_fixed(xq, qlayers, today, m.ACT_MAX), shape, labels)
            print(f'  {short:10s} int16 (today)             {acc16:6.2f} %')
            for label, act_max in (('int8  0..127', 127), ('uint8 0..255', 255)):
                for percentile in (100.0, 99.99):
                    fracs = scales(float_activation_size(data[cal_name][1], layers, percentile), act_max)
                    stats = {'saturated': 0, 'values': 0}
                    acc, classes = accuracy(forward_fixed(xq, qlayers, fracs, act_max, stats), shape, labels)
                    print(f'  {short:10s} {label} p{percentile:<6g}     {acc:6.2f} % ({acc - acc16:+.2f} pp), '
                          f'same class as int16 {100 * (classes == classes16).mean():.2f} %, '
                          f'saturated {100 * stats["saturated"] / stats["values"]:.3f} %')
        print(f'  ({time.time() - start:.0f} s)')


if __name__ == '__main__':
    main()
