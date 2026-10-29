"""What the waveletcnn kernel's host side needs, in numpy only.

preprocess      raw L1a values (120 bands) -> the kernel's input (110 bands, z-scored)
load_weights    weights/quantized/waveletcnn/waveletcnn_int8.npz -> float32 layers
forward         a numpy model of the kernel: patches in, scores out.
classify_fixed  the optimized kernel (waveletcnn_opt) bit for bit: the raw uint16
                cube in, int32 scores out. It preprocesses in integers itself
                (preprocess_fixed), cuts the image into patches like the notebook
                (forward_patch_fixed per patch) and puts the scores back.
                with_class adds the class, as the kernel writes it.

The golden kernel (waveletcnn) does no preprocessing: its notebook does it, the
same way as preprocess here. waveletcnn_opt does it in hardware, in integers, as
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
WEIGHTS = ROOT / 'weights/quantized/waveletcnn/waveletcnn_int8.npz'
DWT_LEN = 8
DWT_PAD = 6
SUB_CH = 58
# waveletcnn_opt's fixed-point formats (the *_FRAC constants in waveletcnn_opt.hpp):
# its input is preprocessing's int8 (IN_FRAC, IN_MAX), its 1D DWT outputs (and shifted
# input bands) have DWT_FRAC = IN_FRAC + 7 = 12 fraction bits (exact, no rounding or
# clipping before projection), its activations between the blocks have ACT_FRAC fraction
# bits (int16), and its scores are int32 with OUT_FRAC.
# LeakyReLU(0.01) multiplies negative values by LEAK_NUM * 2^-LEAK_FRAC = 41 / 4096.
DWT_FRAC, ACT_FRAC, OUT_FRAC = 12, 9, 15
LEAK_NUM, LEAK_FRAC = 41, 12


def load_weights(path=WEIGHTS):
    """(h0, h1, (proj_w, proj_b), (conv_w, conv_b), (head_w, head_b)): q * 2^-frac, as float32."""
    z = np.load(path)
    def dq(name):
        return z[name].astype(np.float32) * np.float32(2.0 ** -int(z[f'{name}.frac']))
    return (dq('dwt_spectral.h0'), dq('dwt_spectral.h1'),
            (dq('projection.weight'), dq('projection.bias')),
            (dq('conv_block.0.weight'), dq('conv_block.0.bias')),
            (dq('seg_head.0.weight'), dq('seg_head.0.bias')))


def _dwt_concat(x, h0, h1):
    """1-level 1D Daubechies-4 DWT along the 110 spectral bands (zero-padded by 6 on each side),
    concatenated onto x: H x W x 110 -> H x W x 226."""
    xp = np.pad(x, ((0, 0), (0, 0), (DWT_PAD, DWT_PAD)))
    f0, f1 = h0[0, 0], h1[0, 0]
    lo = np.stack([xp[:, :, 2 * m:2 * m + DWT_LEN] @ f0 for m in range(SUB_CH)], axis=-1)
    hi = np.stack([xp[:, :, 2 * m:2 * m + DWT_LEN] @ f1 for m in range(SUB_CH)], axis=-1)
    return np.concatenate([x, lo, hi], axis=-1)


def _conv1x1(x, w, b):
    """1 x 1 convolution, x: H x W x IN, w: OUT x IN x 1 x 1."""
    return x @ w[:, :, 0, 0].T + b


def _conv3x3(x, w, b):
    """Same-padded (zero) 3 x 3 convolution, x: H x W x IN, w: OUT x IN x 3 x 3."""
    h, wd, _ = x.shape
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    out = np.zeros((h, wd, w.shape[0]), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += xp[i:i + h, j:j + wd, :] @ w[:, :, i, j].T
    return out + b


def _leaky_relu(x, slope=np.float32(0.01)):
    return np.where(x > 0, x, slope * x)


def forward_patch(x, layers):
    """One H x W x 110 patch -> H x W x 3 scores."""
    h0, h1, (pw, pb), (cw, cb), (hw, hb) = layers
    x = _dwt_concat(x, h0, h1)
    x = _conv1x1(x, pw, pb)
    x = _leaky_relu(_conv3x3(x, cw, cb))
    return _leaky_relu(_conv1x1(x, hw, hb))


def forward(patches, layers):
    """N x H x W x 110 patches -> N x H x W x 3 scores."""
    return np.stack([forward_patch(np.asarray(p, dtype=np.float32), layers) for p in patches])


# ---- the optimized kernel's integer arithmetic ----

def load_quantized(path=WEIGHTS):
    """Quantized weights and their fraction bit counts, as int64 integers."""
    z = np.load(path)
    return (
        (z['dwt_spectral.h0'].astype(np.int64), int(z['dwt_spectral.h0.frac']),
         z['dwt_spectral.h1'].astype(np.int64), int(z['dwt_spectral.h1.frac'])),
        (z['projection.weight'].astype(np.int64), int(z['projection.weight.frac']),
         z['projection.bias'].astype(np.int64), int(z['projection.bias.frac'])),
        (z['conv_block.0.weight'].astype(np.int64), int(z['conv_block.0.weight.frac']),
         z['conv_block.0.bias'].astype(np.int64), int(z['conv_block.0.bias.frac'])),
        (z['seg_head.0.weight'].astype(np.int64), int(z['seg_head.0.weight.frac']),
         z['seg_head.0.bias'].astype(np.int64), int(z['seg_head.0.bias.frac'])),
    )


def _conv3x3_int(x, w, b, bias_shift):
    """_conv3x3 in integers: sum(act * q) + bias * 2^bias_shift, exact in int64."""
    h, wd, _ = x.shape
    xp = np.pad(x.astype(np.int64), ((1, 1), (1, 1), (0, 0)))
    out = np.zeros((h, wd, w.shape[0]), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            out += xp[i:i + h, j:j + wd, :] @ w[:, :, i, j].T
    return out + (b << bias_shift)


def _round_shift(s, shift):
    return s if shift == 0 else (s + (1 << (shift - 1))) >> shift


def _leaky_int(v):
    """LeakyReLU(0.01) in integers: v >= 0 ? v : round_shift(v * LEAK_NUM, LEAK_FRAC)."""
    return np.where(v >= 0, v, _round_shift(v * LEAK_NUM, LEAK_FRAC))


def _requant_act(s, shift):
    """Round off shift bits, saturate to int16 [-32768, 32767]."""
    return np.clip(_round_shift(s, shift), -32768, 32767)


def _requant_leaky_relu(s, shift):
    """Round off shift bits, apply LeakyReLU(0.01), saturate to int16 [-32768, 32767]."""
    return np.clip(_leaky_int(_round_shift(s, shift)), -32768, 32767)


def forward_patch_fixed(xq, qlayers):
    """One H x W x 110 int8 patch -> H x W x 3 int32 scores (score * 2^OUT_FRAC), exactly
    what waveletcnn_opt computes."""
    (h0, fh0, h1, fh1), (pw, fpw, pb, gpb), (cw, fcw, cb, gcb), (hw, fhw, hb, ghb) = qlayers
    if fh0 != fh1:
        raise ValueError('dwt_spectral.h0 and h1 must share the same fraction bit count')
    dwt_frac = IN_FRAC + fh0
    xp = np.pad(xq.astype(np.int64), ((0, 0), (0, 0), (DWT_PAD, DWT_PAD)))
    lo = np.stack([xp[:, :, 2 * m:2 * m + DWT_LEN] @ h0[0, 0] for m in range(SUB_CH)], axis=-1)
    hi = np.stack([xp[:, :, 2 * m:2 * m + DWT_LEN] @ h1[0, 0] for m in range(SUB_CH)], axis=-1)
    cat = np.concatenate([xq.astype(np.int64) << fh0, lo, hi], axis=-1)

    s_proj = dwt_frac + fpw
    proj = _requant_act(cat @ pw[:, :, 0, 0].T + (pb << (s_proj - gpb)), s_proj - ACT_FRAC)

    s_conv = ACT_FRAC + fcw
    act1 = _requant_leaky_relu(_conv3x3_int(proj, cw, cb, s_conv - gcb), s_conv - ACT_FRAC)

    s_head = ACT_FRAC + fhw
    head = act1 @ hw[:, :, 0, 0].T + (hb << (s_head - ghb))
    return _leaky_int(_round_shift(head, s_head - OUT_FRAC)).astype(np.int32)


def classify_fixed(cube, qlayers=None, prep=None):
    """Raw cube (H x W x 120 integers) -> H x W x 3 int32 scores (score * 2^OUT_FRAC), exactly
    what waveletcnn_opt writes: 32 x 32 patches in raster order, the image grown by
    repeating its last row and column."""
    qlayers = qlayers if qlayers is not None else load_quantized()
    prep = prep if prep is not None else prep_constants()
    scores = np.zeros(cube.shape[:2] + (3,), dtype=np.int32)
    for patch, tile in patching.iter_patches(cube, pad_mode='edge'):
        patching.place(scores, forward_patch_fixed(preprocess_fixed(patch, prep), qlayers), tile)
    return scores


def with_class(scores):
    """H x W x 3 int32 scores -> H x W x 4, what waveletcnn_opt writes: the scores,
    then the class (their argmax, the first one on a tie)."""
    return np.concatenate([scores, scores.argmax(-1)[..., None].astype(np.int32)], axis=-1)
