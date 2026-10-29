"""SpecPW-UNet: the golden modules, the golden kernels and the integer model.

The numpy model (tools/specpw_unet_model.py) is the trained network in float, checked
here against the labelled aegean pixels. Against it, compiled with plain g++:
 - the float golden model (src/golden/specpw_unet_golden.hpp), from the golden modules
   (spectral stem, conv block, up block, maxpool, the skip connection's concat);
 - the golden kernels sp_unet_small/medium/large (int8 weights, float activations), on
   real 32 x 32 patches;
 - the integer model's plain-loop C++ (src/golden/specpw_unet_fixed_golden.hpp), bit for
   bit against classify_fixed's arithmetic, so it can be an optimized kernel's reference.
Then the optimized kernels sp_unet_SIZE_opt (Winograd, block order, integer fixed point),
compiled with plain g++, bit for bit against classify_fixed on a real raw crop that is not
whole patches. The board notebooks, shared by every model, are tests/test_notebooks.py's.
"""
import contextlib
import gc
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which('g++') or shutil.which('clang++')
PATCH = 32
# two classes in the first patch, all three in the second
CORNERS = ((0, 0), (40, 104))
AP_CTRL, DIN, DOUT, N_REG = 0x00, 0x10, 0x1c, 0x28
OPT_DESIGN = ROOT / 'src/hls/sp_unet_small_opt/sp_unet_opt.hpp'
DINS, OPT_DOUT, HEIGHT, WIDTH, CYCLES = (0x10, 0x1c, 0x28, 0x34), 0x40, 0x4c, 0x54, 0x5c


FLOAT_DRIVER = """
#include "specpw_unet_golden.hpp"
#include <cstdio>

typedef specpw_unet_params_golden<float, 110, 3, specpw_unet_{size}> params_t;
static params_t params;
static float din[{n}][{n}][110], dout[{n}][{n}][3];

int main(int, char** argv) {{
    // every member is a float array, so the struct is the state_dict's floats in order
    FILE* f = fopen(argv[1], "rb");
    if (fread(&params, 1, sizeof params, f) != sizeof params || fgetc(f) != EOF) return 1;
    fclose(f);
    f = fopen(argv[2], "rb");
    if (fread(din, 1, sizeof din, f) != sizeof din) return 1;
    fclose(f);
    specpw_unet_golden<float, {n}, {n}, 110, 3, specpw_unet_{size}>(din, params, 1e-5f, dout);
    f = fopen(argv[3], "wb");
    fwrite(dout, 1, sizeof dout, f);
    fclose(f);
}}
"""

KERNEL_DRIVER = """
#include "{name}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int n = atoi(argv[3]);
    std::vector<float> din(n * 32 * 32 * 110), dout(n * 32 * 32 * 3);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(float), din.size(), f) != din.size()) return 1;
    fclose(f);
    {name}(din.data(), dout.data(), n);
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(float), dout.size(), f); fclose(f);
}}
"""

FIXED_DRIVER = """
#include "specpw_unet_fixed_golden.hpp"
#include "{weights}"
#include <cstdio>
#include <cstdlib>

typedef specpw_unet_quantized_params_golden<110, 3, specpw_unet_{size}> qparams_t;
static qparams_t params;
static int32_t din[32][32][110], dout[32][32][3];

int main(int, char** argv) {{
    SPECPW_UNET_LOAD_QUANTIZED_PARAMS(params, sp_unet_{size}_int8);
    const int n = atoi(argv[3]), act_max = atoi(argv[4]);
    FILE* in = fopen(argv[1], "rb");
    FILE* out = fopen(argv[2], "wb");
    for (int p = 0; p < n; p++) {{
        if (fread(din, 1, sizeof din, in) != sizeof din) return 1;
        specpw_unet_fixed_golden<32, 32, 110, 3, specpw_unet_{size}>(din, params, {in_frac}, {act_frac}, {out_frac}, act_max, dout);
        fwrite(dout, 1, sizeof dout, out);
    }}
    fclose(in); fclose(out);
}}
"""


OPT_DRIVER = """
#include "{name}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    std::vector<uint16_t> cube((size_t)h * w * CUBE_BANDS);
    FILE* f = fopen(argv[1], "rb");
    if (fread(cube.data(), 2, cube.size(), f) != cube.size()) return 1;
    fclose(f);
    std::vector<cube_word_t> din(cube.size() / BANDS_PER_WORD);
    for (size_t i = 0; i < cube.size(); i++) din[i / BANDS_PER_WORD][i % BANDS_PER_WORD] = cube[i];
    std::vector<score_word_t> dout((size_t)h * w);
    uint32_t cycles = 0;
    {name}({ports}dout.data(), h, w, cycles);
    f = fopen(argv[2], "wb");
    for (auto &o : dout) for (int c = 0; c < 4; c++) {{ const int32_t v = o[c]; fwrite(&v, 4, 1, f); }}
    fclose(f);
}}
"""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_and_run(source_text, tmp, stem, include_dirs, sources, args):
    source = tmp / f'{stem}.cpp'
    source.write_text(source_text)
    exe = tmp / f'{stem}.exe'
    subprocess.run([CXX, '-std=c++14', '-O2', *[f'-I{d}' for d in include_dirs], str(source),
                    *[str(s) for s in sources], '-o', str(exe)], check=True)
    subprocess.run([str(exe), *[str(a) for a in args]], check=True)


@unittest.skipUnless(np is not None, 'needs numpy')
class SpecPWUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('specpw_unet_model', ROOT / 'tools/specpw_unet_model.py')
        cls.x, cls.labels = cls.model.aegean_image()
        cls.raw, _ = cls.model.aegean_raw()
        cls.weights = {size: cls.model.load_weights(size) for size in cls.model.SIZES}

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def quantized(self, size):
        if not self.model.quantized_path(size).exists():
            self.skipTest(f'no quantized weights: python tools/quantize_weights.py weights/fp32/sp_unet_{size}')
        return self.model.load_quantized(size)

    def patches(self, image):
        return np.stack([image[i:i + PATCH, j:j + PATCH] for i, j in CORNERS])

    def test_numpy_model_classifies_the_labelled_pixels(self):
        """stem, U-Net and skip in the trained order: about the test accuracy training
        reported (95 %)."""
        for size in self.model.SIZES:
            with self.subTest(size=size):
                scores = self.model.forward(self.x, self.weights[size])
                self.assertGreater((scores.argmax(-1) == self.labels).mean(), 0.93)

    def test_skip_comes_after_the_upsampled_map(self):
        """cat([skip, upsampled]) instead: the model falls apart."""
        model = self.model
        original = np.concatenate
        try:
            model.np.concatenate = lambda arrays, axis: original(arrays[::-1], axis=axis)
            scores = model.forward(self.x, self.weights['small'])
        finally:
            model.np.concatenate = original
        self.assertLess((scores.argmax(-1) == self.labels).mean(), 0.6)

    def test_int8_weights_and_fixed_point_keep_the_classes(self):
        """Folding BatchNorm and int8 weights, then integer activations: each step changes
        under 0.5 % of the classes and keeps the accuracy, and no activation saturates."""
        for size in self.model.SIZES:
            with self.subTest(size=size):
                qlayers = self.quantized(size)
                fp32 = self.model.forward(self.x, self.weights[size]).argmax(-1)
                golden = self.model.forward_patch(self.x, self.model.dequantize(qlayers)).argmax(-1)
                stats = {}
                xq = self.model.preprocessing.preprocess_fixed(self.raw)
                fixed = self.model.forward_patch_fixed(xq, qlayers, stats).argmax(-1)
                self.assertGreater((golden == fp32).mean(), 0.995)
                self.assertGreater((fixed == golden).mean(), 0.995)
                self.assertGreater((fixed == self.labels).mean(), 0.93)
                self.assertEqual(stats['saturated'], 0)
                self.assertLess(stats['max_sum'], 2 ** 31, 'an int32 accumulator must hold every sum')

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_cpp_golden_matches_the_numpy_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for size in self.model.SIZES:
                with self.subTest(size=size):
                    self.model.flat_weights(self.weights[size]).tofile(tmp / 'weights.bin')
                    for k, patch in enumerate(self.patches(self.x).astype(np.float32)):
                        patch.tofile(tmp / 'in.bin')
                        compile_and_run(FLOAT_DRIVER.format(size=size, n=PATCH), tmp, f'float_{size}',
                                        [ROOT / 'src/golden'], [],
                                        [tmp / 'weights.bin', tmp / 'in.bin', tmp / 'out.bin']) if k == 0 else \
                            subprocess.run([str(tmp / f'float_{size}.exe'), str(tmp / 'weights.bin'),
                                            str(tmp / 'in.bin'), str(tmp / 'out.bin')], check=True)
                        cpp = np.fromfile(tmp / 'out.bin', dtype=np.float32).reshape(PATCH, PATCH, 3)
                        want = self.model.forward(patch, self.weights[size])
                        np.testing.assert_allclose(cpp, want, rtol=1e-4, atol=1e-3)
                        self.assertTrue((cpp.argmax(-1) == want.argmax(-1)).all())

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_golden_kernels_match_the_numpy_model_on_real_patches(self):
        """Two patches in one start: each at its own place in DDR."""
        patches = self.patches(self.x).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            patches.tofile(tmp / 'in.bin')
            for size in self.model.SIZES:
                with self.subTest(size=size):
                    layers = self.model.dequantize(self.quantized(size))
                    name = self.model.name(size)
                    source_dir = ROOT / 'src/hls' / name
                    compile_and_run(KERNEL_DRIVER.format(name=name), tmp, f'run_{name}',
                                    [source_dir, ROOT / 'src/common'], [source_dir / f'{name}.cpp'],
                                    [tmp / 'in.bin', tmp / 'out.bin', len(patches)])
                    cpp = np.fromfile(tmp / 'out.bin', dtype=np.float32).reshape(len(patches), PATCH, PATCH, 3)
                    for got, patch in zip(cpp, patches):
                        want = self.model.forward_patch(patch, layers)
                        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
                        self.assertTrue((got.argmax(-1) == want.argmax(-1)).all())

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_fixed_point_cpp_matches_the_numpy_model_bit_for_bit(self):
        """The real patches, and one of random int8 values up to +-127; with int16's
        ceiling, and again with a ceiling of 2.0 (2047), so the saturation is exercised
        (no aegean activation reaches int16's)."""
        m = self.model
        real = m.preprocessing.preprocess_fixed(self.patches(self.raw)).astype(np.int32)
        noise = np.random.default_rng(5).integers(-127, 128, (1, PATCH, PATCH, 110)).astype(np.int32)
        inputs = np.concatenate([real, noise])
        saturated = False
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inputs.tofile(tmp / 'in.bin')
            for size in m.SIZES:
                with self.subTest(size=size):
                    qlayers = self.quantized(size)
                    weights = m.quantized_path(size).with_suffix('.hpp')
                    compile_and_run(FIXED_DRIVER.format(size=size, weights=weights.as_posix(), in_frac=m.IN_FRAC,
                                                        act_frac=m.ACT_FRAC, out_frac=m.OUT_FRAC),
                                    tmp, f'fixed_{size}', [ROOT / 'src/golden'], [],
                                    [tmp / 'in.bin', tmp / 'out.bin', len(inputs), m.ACT_MAX])
                    for act_max in (m.ACT_MAX, 2047):
                        subprocess.run([str(tmp / f'fixed_{size}.exe'), str(tmp / 'in.bin'), str(tmp / 'out.bin'),
                                        str(len(inputs)), str(act_max)], check=True)
                        cpp = np.fromfile(tmp / 'out.bin', dtype=np.int32).reshape(len(inputs), PATCH, PATCH, 3)
                        for got, xq in zip(cpp, inputs):
                            stats = {}
                            np.testing.assert_array_equal(got, m.forward_patch_fixed(xq, qlayers, stats, act_max))
                            saturated |= stats['saturated'] > 0
        self.assertTrue(saturated, 'no activation saturated: the clipping was not tested')

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_optimized_kernels_match_classify_fixed_bit_for_bit(self):
        """The whole raw-cube kernels, Winograd (_opt), direct (_opt_direct) and the Zynq-7020's, on a real crop of
        45 x 70 pixels (2 x 3 patches, the edge ones partly outside): what classify_fixed and
        the class say, every value."""
        m = self.model
        cube = np.clip(np.rint(self.raw[:45, :70]), 0, 65535).astype(np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cube.tofile(tmp / 'cube.bin')
            for name, size in [(f'{m.name(size)}_opt{kind}', size) for size in m.SIZES
                               for kind in ('', '_direct', '_direct_zynq7020')
                               if kind != '_direct_zynq7020' or size == 'small']:   # the 7020 fits small only
                with self.subTest(kernel=name):
                    source_dir = ROOT / 'src/hls' / name
                    ports = 'din.data(), ' * (1 if name.endswith('zynq7020') else 4)   # read ports
                    compile_and_run(OPT_DRIVER.format(name=name, ports=ports), tmp, f'run_{name}',
                                    [source_dir, ROOT / 'src/common'], sorted(source_dir.glob('*.cpp')),
                                    [tmp / 'cube.bin', tmp / 'out.bin', 45, 70])
                    got = np.fromfile(tmp / 'out.bin', dtype=np.int32).reshape(45, 70, 4)
                    want = m.kernel_reference(cube, self.quantized(size))
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(len(np.unique(got[..., 3])), 3, 'all three classes occur')

    def test_classify_fixed_cuts_the_image_like_the_notebook(self):
        """classify_fixed on a raw cube that is not whole patches: each patch's answer,
        the edge ones grown by repeating the last row and column."""
        m = self.model
        qlayers = self.quantized('small')
        cube = np.clip(np.rint(self.raw[:40, :50]), 0, 65535).astype(np.uint16)
        scores = m.classify_fixed(cube, qlayers)
        grown = np.pad(cube, ((0, 24), (0, 14), (0, 0)), mode='edge')
        prep = m.preprocessing.prep_constants()
        for i in (0, 32):
            for j in (0, 32):
                want = m.forward_patch_fixed(m.preprocessing.preprocess_fixed(
                    grown[i:i + PATCH, j:j + PATCH], prep), qlayers)
                np.testing.assert_array_equal(scores[i:i + PATCH, j:j + PATCH], want[:40 - i, :50 - j])


if __name__ == '__main__':
    unittest.main()
