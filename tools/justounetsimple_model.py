"""What the justounetsimple kernel's host side needs, in numpy only.

preprocess      raw L1a values (120 bands) -> the kernel's input (110 bands, z-scored)
load_weights    weights/quantized/justounetsimple/justounetsimple_int8.npz -> float32 layers
forward         a numpy model of the kernel: patches in, scores out.
classify_fixed  the optimized kernel (justounetsimple_opt) bit for bit: the raw uint16
                cube in, int32 scores out. It preprocesses in integers itself
                (preprocess_fixed), cuts the image into patches like the notebook
                (forward_patch_fixed per patch) and puts the scores back.
                with_class adds the class, as the kernel writes it.

The golden kernel (justounetsimple) does no preprocessing: its notebook does it, the
same way as preprocess here. justounetsimple_opt does it in hardware, in integers, as
preprocess_fixed here. Both come from tools/preprocessing.py, which every model shares.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patching  # noqa: E402
from preprocessing import (MU_SD, IN_FRAC, IN_MAX, read_preprocessing, preprocess,  # noqa: E402,F401
                           quantize_input, prep_constants, preprocess_fixed)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'weights/quantized/justounetsimple/justounetsimple_int8.npz'
# justounetsimple_opt's fixed-point formats (the *_FRAC constants in justounetsimple_opt.hpp):
# its input is preprocessing's int8 (IN_FRAC, IN_MAX), its activations have ACT_FRAC
# fraction bits (int16), its scores are int32 with OUT_FRAC.
ACT_FRAC, OUT_FRAC = 11, 16


def load_weights(path=WEIGHTS):
    """[(weight, bias)] for conv1..conv4: q * 2^-frac, as float32."""
    z = np.load(path)
    layers = []
    for n in range(1, 5):
        w = z[f'conv{n}.weight'].astype(np.float32) * np.float32(2.0 ** -int(z[f'conv{n}.weight.frac']))
        b = z[f'conv{n}.bias'].astype(np.float32) * np.float32(2.0 ** -int(z[f'conv{n}.bias.frac']))
        layers.append((w, b))
    return layers


def _conv(x, w, b):
    """Same-padded (zero) 3 x 3 convolution, x: H x W x IN, w: OUT x IN x 3 x 3."""
    h, wd, _ = x.shape
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    out = np.zeros((h, wd, w.shape[0]), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += xp[i:i + h, j:j + wd, :] @ w[:, :, i, j].T
    return out + b


def _pool(x):
    h, w, c = x.shape
    return x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


def _up(x):
    return x.repeat(2, axis=0).repeat(2, axis=1)


def forward_patch(x, layers):
    """One H x W x 110 patch -> H x W x 3 scores (H, W divisible by 4)."""
    (w1, b1), (w2, b2), (w3, b3), (w4, b4) = layers
    x = _pool(np.maximum(_conv(x, w1, b1), 0))                 # block A
    x = _up(_pool(np.maximum(_conv(x, w2, b2), 0)))            # block B
    x = _up(np.maximum(_conv(x, w3, b3), 0))                   # block C
    return _conv(x, w4, b4)                                    # block D, no ReLU


def forward(patches, layers):
    """N x H x W x 110 patches -> N x H x W x 3 scores."""
    return np.stack([forward_patch(np.asarray(p, dtype=np.float32), layers) for p in patches])


# ---- the optimized kernel's integer arithmetic ----

def load_quantized(path=WEIGHTS):
    """[(q weight, weight frac, q bias, bias frac)] for conv1..conv4, as int64 integers."""
    z = np.load(path)
    return [(z[f'conv{n}.weight'].astype(np.int64), int(z[f'conv{n}.weight.frac']),
             z[f'conv{n}.bias'].astype(np.int64), int(z[f'conv{n}.bias.frac'])) for n in range(1, 5)]


def _conv_int(x, w, b, bias_shift):
    """_conv in integers: sum(act * q) + bias * 2^bias_shift, exact in int64."""
    h, wd, _ = x.shape
    xp = np.pad(x.astype(np.int64), ((1, 1), (1, 1), (0, 0)))
    out = np.zeros((h, wd, w.shape[0]), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            out += xp[i:i + h, j:j + wd, :] @ w[:, :, i, j].T
    return out + (b << bias_shift)


def _round_shift(s, shift):
    return s if shift == 0 else (s + (1 << (shift - 1))) >> shift


def _requant_relu(s, shift):
    """ReLU, round off shift bits, saturate at the int16 maximum."""
    return np.clip(_round_shift(s, shift), 0, 32767)


def forward_patch_fixed(xq, qlayers):
    """One H x W x 110 int8 patch -> H x W x 3 int32 scores (score * 2^OUT_FRAC), exactly
    what justounetsimple_opt computes."""
    (w1, f1, b1, g1), (w2, f2, b2, g2), (w3, f3, b3, g3), (w4, f4, b4, g4) = qlayers
    s1, s2, s3, s4 = IN_FRAC + f1, ACT_FRAC + f2, ACT_FRAC + f3, ACT_FRAC + f4  # fraction bits of the sums
    x = _pool(_requant_relu(_conv_int(xq, w1, b1, s1 - g1), s1 - ACT_FRAC))           # block A
    x = _up(_pool(_requant_relu(_conv_int(x, w2, b2, s2 - g2), s2 - ACT_FRAC)))       # block B
    x = _up(_requant_relu(_conv_int(x, w3, b3, s3 - g3), s3 - ACT_FRAC))              # block C
    return _round_shift(_conv_int(x, w4, b4, s4 - g4), s4 - OUT_FRAC).astype(np.int32)  # block D


def classify_fixed(cube, qlayers=None, prep=None):
    """Raw cube (H x W x 120 integers) -> H x W x 3 int32 scores (score * 2^OUT_FRAC), exactly
    what justounetsimple_opt writes: 32 x 32 patches in raster order, the image grown by
    repeating its last row and column."""
    qlayers = qlayers if qlayers is not None else load_quantized()
    prep = prep if prep is not None else prep_constants()
    scores = np.zeros(cube.shape[:2] + (3,), dtype=np.int32)
    for patch, tile in patching.iter_patches(cube, pad_mode='edge'):
        patching.place(scores, forward_patch_fixed(preprocess_fixed(patch, prep), qlayers), tile)
    return scores


def with_class(scores):
    """H x W x 3 int32 scores -> H x W x 4, what justounetsimple_opt writes: the scores,
    then the class (their argmax, the first one on a tie)."""
    return np.concatenate([scores, scores.argmax(-1)[..., None].astype(np.int32)], axis=-1)
