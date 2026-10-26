"""SpecPW-UNet (sp_unet_small, sp_unet_medium, sp_unet_large) in numpy only.

The trained model, in float:
load_weights   weights/fp32/sp_unet_SIZE/*.pt -> {state_dict name: float32 array}
forward        BatchNorm as trained (not folded): a preprocessed (H, W, 110) image in,
               (H, W, 3) scores out. H and W must be divisible by 4. The reference
               src/golden/specpw_unet_golden.hpp is checked against.
flat_weights   every tensor in state_dict order as one float32 vector: the layout of
               specpw_unet_params_golden, so a testbench can read it in one go.

The golden kernels (src/hls/sp_unet_SIZE/), int8 weights and float activations:
load_quantized weights/quantized/sp_unet_SIZE/sp_unet_SIZE_int8.npz -> {layer: (q weight,
               weight frac, q bias, bias frac)}, BatchNorm folded in
forward_patch  what the kernel computes for one preprocessed 32 x 32 x 110 patch, in
               float32 (the weights q * 2^-frac); the same to float rounding
classify       a raw cube (H x W x 120) -> its scores, cut into 32 x 32 patches like the
               notebook (the last row and column repeated)

The integer arithmetic of the optimized kernels (sp_unet_SIZE_opt), bit for bit (the
C++ plain-loop version is src/golden/specpw_unet_fixed_golden.hpp, and the two agree
exactly, tests/test_specpw_unet.py):
forward_patch_fixed  one int8 patch (z-score * 2^IN_FRAC) -> int32 scores * 2^OUT_FRAC
classify_fixed       a raw cube -> int32 scores, with justounetsimple_opt's integer
                     z-score (justounetsimple_model.preprocess_fixed)
kernel_reference     a raw cube -> what sp_unet_SIZE_opt writes (H x W x 4 int32)

    python tools/specpw_unet_model.py                    # each size on the aegean pixels
    python tools/specpw_unet_model.py reference IMAGE    # IMAGE/reference_sp_unet_SIZE.npy
    python tools/specpw_unet_model.py fixed IMAGE        # IMAGE/reference_fixed_sp_unet_SIZE.npy

reference writes, for an image folder made by tools/justounetsimple_image.py prepare, what
each golden kernel will answer for its cube.npy (float32 H x W x 3): the notebook compares
the board with it. fixed writes what each optimized kernel writes, bit for bit: the scores
* 2^OUT_FRAC, then the class (int32 H x W x 4); its cube must be raw counts (uint16).

The skip connection is cat([upsampled, skip]): on the aegean pixels that order gives
94-96 % and the other 27-40 %. The aegean pixels are every 8th of the capture, so the
models see coarser images than they were trained on; the accuracy is a sanity check,
not the models' score.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_weights  # noqa: E402
import justounetsimple_model  # noqa: E402
import patching  # noqa: E402

SIZES = ('small', 'medium', 'large')
EPS = 1e-5
PATCH = 32
# The convs in order, as the quantized weights name them (BatchNorm folded in).
LAYERS = ('convsp1', 'convsp2', 'convsp3', 'double1.block.0', 'double1.block.3',
          'double2.block.0', 'double2.block.3', 'up1.1', 'double3.block.0', 'double3.block.3',
          'up2.1', 'double4.block.0', 'double4.block.3', 'final')

# The integer formats (src/golden/specpw_unet_fixed_golden.hpp takes them as arguments):
# the input is the z-score * 2^IN_FRAC as int8, saturated at +-127 (justounetsimple_opt's
# preprocessing); every activation int16 with ACT_FRAC fraction bits, ReLU'd and saturated
# at 32767 (up to 32.0); the scores int32 with OUT_FRAC. On the aegean pixels the largest
# activation is 9.1 (large's double1), so 3.5 times that still fits. Each conv sums
# act * q exactly, its bias shifted up to the sum's fraction bits, and rounds half up
# (adds half, shifts right) to the next layer's.
IN_FRAC, ACT_FRAC, OUT_FRAC = justounetsimple_model.IN_FRAC, 10, 16
ACT_MAX = 32767


def name(size):
    return f'sp_unet_{size}'


def checkpoint(size):
    return next((ROOT / 'weights/fp32' / name(size)).glob('*.pt'))


def load_weights(size):
    """{state_dict name: float32 array}, without num_batches_tracked, in state_dict order."""
    _, _, tensors = extract_weights.load(checkpoint(size))
    return tensors


def flat_weights(tensors):
    return np.concatenate([t.reshape(-1) for t in tensors.values()]).astype(np.float32)


def _conv(x, w, b=None):
    """Same-padded conv, stride 1. x (H, W, in), w (out, in, k, k)."""
    k = w.shape[2]
    pad = (k - 1) // 2
    h, wd = x.shape[:2]
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    out = np.zeros((h, wd, w.shape[0]), dtype=np.result_type(x, w))
    for i in range(k):
        for j in range(k):
            out += xp[i:i + h, j:j + wd] @ w[:, :, i, j].T
    return out if b is None else out + b


def _bn(x, t, layer):
    return ((x - t[f'{layer}.running_mean']) / np.sqrt(t[f'{layer}.running_var'] + EPS)
            * t[f'{layer}.weight'] + t[f'{layer}.bias'])


def _relu(x):
    return np.maximum(x, 0)


def _pool(x):
    h, w, c = x.shape
    return x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


def _up(x):
    return x.repeat(2, axis=0).repeat(2, axis=1)


def _conv_block(x, t, layer):
    x = _relu(_bn(_conv(x, t[f'{layer}.block.0.weight']), t, f'{layer}.block.1'))
    return _relu(_bn(_conv(x, t[f'{layer}.block.3.weight']), t, f'{layer}.block.4'))


def _up_block(x, t, layer):
    return _relu(_bn(_conv(_up(x), t[f'{layer}.1.weight']), t, f'{layer}.2'))


def forward(x, t):
    """Scores (H, W, classes) of a preprocessed image x (H, W, 110), in float64."""
    t = {k: v.astype(np.float64) for k, v in t.items()}
    x = np.asarray(x, dtype=np.float64)
    x = _relu(_bn(_conv(x, t['convsp1.weight'], t['convsp1.bias']), t, 'bnsp1'))
    x = _relu(_conv(x, t['convsp2.weight'], t['convsp2.bias']))
    x = _relu(_conv(x, t['convsp3.weight'], t['convsp3.bias']))
    skip = _conv_block(x, t, 'double1')
    x = _pool(_conv_block(_pool(skip), t, 'double2'))
    x = _conv_block(_up_block(x, t, 'up1'), t, 'double3')
    x = _up_block(x, t, 'up2')
    x = _conv_block(np.concatenate([x, skip], axis=-1), t, 'double4')
    return _conv(x, t['final.weight'], t['final.bias'])


def _network(x, conv_relu, final):
    """The quantized network's order of layers, for both arithmetics: conv_relu(x, layer)
    is a conv with its ReLU, final(x) the last 1x1 conv."""
    for layer in LAYERS[:3]:                                          # spectral stem
        x = conv_relu(x, layer)
    skip = conv_relu(conv_relu(x, 'double1.block.0'), 'double1.block.3')
    x = conv_relu(conv_relu(_pool(skip), 'double2.block.0'), 'double2.block.3')
    x = conv_relu(_up(_pool(x)), 'up1.1')
    x = conv_relu(conv_relu(x, 'double3.block.0'), 'double3.block.3')
    x = conv_relu(_up(x), 'up2.1')
    x = np.concatenate([x, skip], axis=-1)                            # skip connection
    x = conv_relu(conv_relu(x, 'double4.block.0'), 'double4.block.3')
    return final(x)


# ---- the golden kernels: int8 weights, float activations ----

def quantized_path(size):
    return ROOT / 'weights/quantized' / name(size) / f'{name(size)}_int8.npz'


def load_quantized(size):
    """{layer: (q weight, weight frac, q bias, bias frac)}, as int64 integers."""
    z = np.load(quantized_path(size))
    return {layer: (z[f'{layer}.weight'].astype(np.int64), int(z[f'{layer}.weight.frac']),
                    z[f'{layer}.bias'].astype(np.int64), int(z[f'{layer}.bias.frac'])) for layer in LAYERS}


def dequantize(qlayers):
    """{layer: (weight, bias)}: q * 2^-frac, as float32."""
    return {layer: (w.astype(np.float32) * np.float32(2.0 ** -wf), b.astype(np.float32) * np.float32(2.0 ** -bf))
            for layer, (w, wf, b, bf) in qlayers.items()}


def forward_patch(x, layers):
    """One preprocessed H x W x 110 patch -> H x W x 3 scores, in float32: the golden kernel."""
    x = np.asarray(x, dtype=np.float32)
    return _network(x, lambda h, layer: _relu(_conv(h, *layers[layer])),
                    lambda h: _conv(h, *layers['final']))


def classify(cube, layers):
    """Raw cube (H x W x 120) -> H x W x 3 float32 scores, what the golden kernel writes: the
    float preprocessing, then 32 x 32 patches in raster order, the image grown by repeating
    its last row and column."""
    kept, mean, inv_std = justounetsimple_model.read_preprocessing()
    scores = np.zeros(cube.shape[:2] + (3,), dtype=np.float32)
    for patch, tile in patching.iter_patches(cube, PATCH, pad_mode='edge'):
        x = justounetsimple_model.preprocess(patch, kept, mean, inv_std)
        patching.place(scores, forward_patch(x, layers), tile, PATCH)
    return scores


# ---- the integer arithmetic, bit for bit ----

def _round_shift(s, shift):
    """s * 2^-shift rounded half up; a negative shift is exact (shifts left)."""
    if shift <= 0:
        return s << -shift
    return (s + (1 << (shift - 1))) >> shift


def _conv_fixed(x, qlayer, in_frac):
    """The conv's exact sum, int64, with sum_frac = in_frac + weight frac fraction bits."""
    w, wf, b, bf = qlayer
    sum_frac = in_frac + wf
    if sum_frac < bf:
        raise ValueError(f'the bias has more fraction bits ({bf}) than the sum ({sum_frac})')
    return _conv(x.astype(np.int64), w) + (b << (sum_frac - bf)), sum_frac


def forward_patch_fixed(xq, qlayers, stats=None, act_max=ACT_MAX):
    """One H x W x 110 int8 patch (z-score * 2^IN_FRAC) -> H x W x 3 int32 scores
    (score * 2^OUT_FRAC). stats, a dict, gets the largest |sum| and how many activations
    saturated; act_max lower than int16's only to test the saturation."""
    in_frac = {'convsp1': IN_FRAC}

    def track(s):
        if stats is not None:
            stats['max_sum'] = max(stats.get('max_sum', 0), int(np.abs(s).max()))

    def conv_relu(x, layer):
        s, sum_frac = _conv_fixed(x, qlayers[layer], in_frac.get(layer, ACT_FRAC))
        track(s)
        y = np.maximum(_round_shift(s, sum_frac - ACT_FRAC), 0)
        if stats is not None:
            stats['saturated'] = stats.get('saturated', 0) + int((y > act_max).sum())
        return np.minimum(y, act_max)

    def final(x):
        s, sum_frac = _conv_fixed(x, qlayers['final'], ACT_FRAC)
        track(s)
        return _round_shift(s, sum_frac - OUT_FRAC).astype(np.int32)

    return _network(np.asarray(xq, dtype=np.int64), conv_relu, final)


def classify_fixed(cube, qlayers, prep=None):
    """Raw cube (H x W x 120 integers) -> H x W x 3 int32 scores (score * 2^OUT_FRAC): the
    integer z-score, then 32 x 32 patches as in classify."""
    prep = prep if prep is not None else justounetsimple_model.prep_constants()
    scores = np.zeros(cube.shape[:2] + (3,), dtype=np.int32)
    for patch, tile in patching.iter_patches(cube, PATCH, pad_mode='edge'):
        xq = justounetsimple_model.preprocess_fixed(patch, prep)
        patching.place(scores, forward_patch_fixed(xq, qlayers), tile, PATCH)
    return scores


# ---- images ----

def aegean_image():
    """(preprocessed (72, 136, 110), labels (72, 136)): the labelled aegean capture,
    cut to a multiple of 4."""
    raw, labels = aegean_raw()
    kept, mean, inv_std = justounetsimple_model.read_preprocessing()
    return justounetsimple_model.preprocess(raw, kept, mean, inv_std), labels


def aegean_raw():
    """(raw (72, 136, 120), labels (72, 136)) of the labelled aegean capture."""
    shape = (75, 137)
    raw = np.loadtxt(ROOT / 'aegean/pixels.txt').reshape(*shape, -1)
    labels = np.load(ROOT / 'aegean/labels.npy').reshape(shape)
    return raw[:72, :136], labels[:72, :136]


def reference(image, sizes=SIZES):
    """IMAGE/reference_sp_unet_SIZE.npy for each size: classify on IMAGE/cube.npy."""
    image = Path(image)
    cube = np.load(image / 'cube.npy', mmap_mode='r')
    written = []
    for size in sizes:
        path = image / f'reference_{name(size)}.npy'
        np.save(path, classify(cube, dequantize(load_quantized(size))))
        written.append(path)
    return written


def kernel_reference(cube, qlayers, prep=None):
    """What sp_unet_SIZE_opt writes for a raw cube: H x W x 4 int32, the scores * 2^OUT_FRAC,
    then the class (their argmax, the first one on a tie)."""
    return justounetsimple_model.with_class(classify_fixed(cube, qlayers, prep))


def fixed_reference(image, sizes=SIZES):
    """IMAGE/reference_fixed_sp_unet_SIZE.npy for each size: kernel_reference on IMAGE/cube.npy."""
    image = Path(image)
    cube = np.load(image / 'cube.npy', mmap_mode='r')
    if cube.dtype != np.uint16:
        raise SystemExit(f'{image / "cube.npy"} is {cube.dtype}, not raw counts (uint16): '
                         'make it again with tools/justounetsimple_image.py prepare')
    prep = justounetsimple_model.prep_constants()
    written = []
    for size in sizes:
        path = image / f'reference_fixed_{name(size)}.npy'
        np.save(path, kernel_reference(cube, load_quantized(size), prep))
        written.append(path)
    return written


def main():
    if len(sys.argv) == 3 and sys.argv[1] in ('reference', 'fixed'):
        for path in (reference if sys.argv[1] == 'reference' else fixed_reference)(sys.argv[2]):
            print(f'wrote {path}')
        return
    if len(sys.argv) > 1:
        raise SystemExit(__doc__)
    x, labels = aegean_image()
    raw, _ = aegean_raw()
    for size in SIZES:
        fp32 = forward(x, load_weights(size)).argmax(-1)
        line = f'{name(size)}: fp32 {100 * (fp32 == labels).mean():.2f} %'
        if quantized_path(size).exists():
            qlayers = load_quantized(size)
            golden = forward_patch(x, dequantize(qlayers)).argmax(-1)
            stats = {}
            fixed = forward_patch_fixed(justounetsimple_model.preprocess_fixed(raw), qlayers, stats).argmax(-1)
            line += (f', int8 weights {100 * (golden == labels).mean():.2f} %, fixed point '
                     f'{100 * (fixed == labels).mean():.2f} % ({100 * (fixed == golden).mean():.2f} % '
                     f'the same class as int8; largest sum 2^{np.log2(stats["max_sum"]):.1f}, '
                     f'{stats["saturated"]} activations saturated)')
        print(line + ' of the aegean pixels')


if __name__ == '__main__':
    main()
