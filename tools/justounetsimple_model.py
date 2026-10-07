"""What the justounetsimple kernel's host side needs, in numpy only.

preprocess      raw L1a values (120 bands) -> the kernel's input (110 bands, z-scored)
load_weights    weights/quantized/justounetsimple/justounetsimple_int8.npz -> float32 layers
forward         a numpy model of the kernel: patches in, scores out. The reference the
                board's answers are checked against (tools/justounetsimple_image.py prepare).
forward_patch_fixed
                the optimized kernel (justounetsimple_opt) bit for bit: its integer
                arithmetic, int16 patch in (quantize_input), int32 scores out.

The kernel does no preprocessing: the notebook on the board does it, the same way as
here. The same band selection and z-score as justoliunet is assumed (mu_sd.txt, bands
0-7 and 118-119 dropped).
"""
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'weights/quantized/justounetsimple/justounetsimple_int8.npz'
MU_SD = ROOT / 'mu_sd.txt'
RAW_BANDS = 120
DROPPED_BANDS = [0, 1, 2, 3, 4, 5, 6, 7, 118, 119]
# justounetsimple_opt's fixed-point formats (JOPT_*_FRAC in justounetsimple_opt.hpp):
# its input is x * 2^IN_FRAC as int16, its activations have ACT_FRAC fraction bits,
# its scores are int32 with OUT_FRAC.
IN_FRAC, ACT_FRAC, OUT_FRAC = 11, 11, 16


def read_preprocessing(mu_sd=MU_SD):
    """(kept band indices, per-band mean, per-band 1/std) from the training mu_sd.txt."""
    arrays = re.findall(r'\[([^\]]*)\]', Path(mu_sd).read_text())
    if len(arrays) != 2:
        raise ValueError(f'{mu_sd}: expected a mu array and an sd array')
    mean, std = [np.array([float(v) for v in a.replace(',', ' ').split()]) for a in arrays]
    kept = np.array([b for b in range(RAW_BANDS) if b not in DROPPED_BANDS])
    if len(mean) != len(kept) or len(std) != len(kept):
        raise ValueError(f'{mu_sd} has {len(mean)} means and {len(std)} stds; {len(kept)} bands are kept')
    return kept, mean.astype(np.float32), (1.0 / std).astype(np.float32)


def preprocess(raw, kept, mean, inv_std):
    """(x - mean) * (1/std) on the kept bands, in float32. raw: (..., 120)."""
    return (np.asarray(raw, dtype=np.float32)[..., kept] - mean) * inv_std


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


def quantize_input(x):
    """Preprocessed patches (float) -> the optimized kernel's int16 input."""
    return np.clip(np.rint(np.asarray(x, dtype=np.float32) * np.float32(2.0 ** IN_FRAC)),
                   -32768, 32767).astype(np.int16)


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
    """One H x W x 110 int16 patch -> H x W x 3 int32 scores (score * 2^OUT_FRAC), exactly
    what justounetsimple_opt computes."""
    (w1, f1, b1, g1), (w2, f2, b2, g2), (w3, f3, b3, g3), (w4, f4, b4, g4) = qlayers
    s1, s2, s3, s4 = IN_FRAC + f1, ACT_FRAC + f2, ACT_FRAC + f3, ACT_FRAC + f4  # fraction bits of the sums
    x = _pool(_requant_relu(_conv_int(xq, w1, b1, s1 - g1), s1 - ACT_FRAC))           # block A
    x = _up(_pool(_requant_relu(_conv_int(x, w2, b2, s2 - g2), s2 - ACT_FRAC)))       # block B
    x = _up(_requant_relu(_conv_int(x, w3, b3, s3 - g3), s3 - ACT_FRAC))              # block C
    return _round_shift(_conv_int(x, w4, b4, s4 - g4), s4 - OUT_FRAC).astype(np.int32)  # block D
