"""Every network the kernels implement, in one table, and the PC's answers for a cube.

A kernel belongs to the model whose name it starts with: justounetsimple_opt_zynq7020 is
justounetsimple's, sp_unet_small_opt_direct sp_unet_small's. The kernel named like its
model is the golden one, the rest are optimized. main.py packs MODELS' entry with the
board's notebook (kernel.json), so one notebook serves every model: the golden kernels
share software/pynq/shared/classify_golden.ipynb, the optimized ones classify.ipynb.

The board is checked against the image's labels only. The PC's answers, for the tests
(tests/test_notebooks.py): golden_scores, what the golden kernel answers (float32
H x W x 3), and fixed_scores, what every optimized kernel writes, bit for bit (int32
H x W x 4: the scores * 2^out_frac, then the class).

The table needs nothing but Python (main.py reads it); the answers need numpy.
"""
import importlib
from pathlib import Path
import sys

TOOLS = str(Path(__file__).resolve().parent)

# patch: the golden kernel's 2D patch size, None for a 1D (per-pixel) network.
# out_frac: the optimized kernels' scores are score * 2^out_frac (the model's OUT_FRAC).
MODELS = {
    'justounetsimple': {'patch': 32, 'out_frac': 16},
    'waveletcnn': {'patch': 32, 'out_frac': 15},
    'sp_unet_small': {'patch': 32, 'out_frac': 16},
    'sp_unet_medium': {'patch': 32, 'out_frac': 16},
    'sp_unet_large': {'patch': 32, 'out_frac': 16},
    'justoliunet': {'patch': None, 'out_frac': 16},
    'justoliunet_bn': {'patch': None, 'out_frac': 16},
    'spectralpointwise1d': {'patch': None, 'out_frac': 16},
}
CHUNK = 65536   # pixels at a time for the 1D networks: a whole capture at once needs GBs


def model_of(kernel):
    """The model a kernel implements (the longest name it starts with), or None."""
    found = [m for m in MODELS if kernel == m or kernel.startswith(m + '_')]
    return max(found, key=len) if found else None


def _module(name):
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return importlib.import_module(name)


def _per_pixel(cube, fn):
    """fn on (N, 120) raw pixels, CHUNK at a time -> H x W x (its columns)."""
    import numpy as np
    h, w, bands = cube.shape
    pixels = np.asarray(cube).reshape(h * w, bands)
    return np.concatenate([fn(pixels[s:s + CHUNK]) for s in range(0, h * w, CHUNK)]).reshape(h, w, -1)


def golden_forward(model):
    """The golden kernel's network, in numpy: preprocessed float32 units -> their scores. A
    unit is a 32 x 32 x 110 patch (2D, one at a time) or N x 110 pixels (1D)."""
    import numpy as np
    if MODELS[model]['patch'] is None:
        m = _module('spectralpointwise1d_model' if model == 'spectralpointwise1d' else 'justoliunet_model')
        w = m.load_quantized() if model == 'spectralpointwise1d' else m.load_quantized(model)
        return lambda x: m.forward(x, w).astype(np.float32)
    if model.startswith('sp_unet_'):
        m = _module('specpw_unet_model')
        layers = m.dequantize(m.load_quantized(model[len('sp_unet_'):]))
    else:
        m = _module(f'{model}_model')
        layers = m.load_weights()
    return lambda x: m.forward_patch(x, layers)


def golden_scores(model, cube):
    """Raw cube (H x W x 120) -> H x W x 3 float32: what the golden kernel answers, after the
    notebook's float preprocessing (2D: 32 x 32 patches, the image grown by repeating its
    last row and column)."""
    import numpy as np
    prep, forward = _module('preprocessing'), golden_forward(model)
    kept, mean, inv_std = prep.read_preprocessing()
    if MODELS[model]['patch'] is None:
        return _per_pixel(cube, lambda p: forward(prep.preprocess(p, kept, mean, inv_std)))
    patching = _module('patching')
    scores = np.zeros(cube.shape[:2] + (3,), dtype=np.float32)
    for patch, tile in patching.iter_patches(cube, pad_mode='edge'):
        patching.place(scores, forward(prep.preprocess(patch, kept, mean, inv_std)), tile)
    return scores


def fixed_scores(model, cube):
    """Raw cube (H x W x 120 integers) -> H x W x 4 int32: what every optimized kernel of the
    model writes, bit for bit."""
    if model == 'spectralpointwise1d':
        m = _module('spectralpointwise1d_model')
        q = m.load_quantized_int()
        return _per_pixel(cube, lambda p: m.classify_fixed(p, q))
    if MODELS[model]['patch'] is None:
        m = _module('justoliunet_model')
        q = m.load_quantized_int(model)
        return _per_pixel(cube, lambda p: m.classify_fixed(p, model, q))
    if model.startswith('sp_unet_'):
        m = _module('specpw_unet_model')
        return m.kernel_reference(cube, m.load_quantized(model[len('sp_unet_'):]))
    m = _module(f'{model}_model')
    return m.with_class(m.classify_fixed(cube))
