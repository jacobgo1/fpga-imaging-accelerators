"""SpecPW-UNet: the golden modules, the golden kernels, the integer model and the notebook.

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
whole patches; and the `fixed` command, which writes what they must answer.
The board notebooks' code cells run against a fake `pynq` whose kernel computes the
patches with the numpy model: that checks the register traffic, the patch cutting and
stitching, the preprocessing and the comparison -- not PYNQ or the board.
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
NOTEBOOK = ROOT / 'software/pynq/sp_unet_small/sp_unet_small.ipynb'
AP_CTRL, DIN, DOUT, N_REG = 0x00, 0x10, 0x1c, 0x28
OPT_NOTEBOOK = ROOT / 'software/pynq/sp_unet_small_opt/sp_unet_small_opt.ipynb'
OPT_DESIGN = ROOT / 'src/hls/sp_unet_small_opt/sp_unet_opt.hpp'
DINS, OPT_DOUT, HEIGHT, WIDTH, CYCLES = (0x10, 0x1c, 0x28, 0x34), 0x40, 0x4c, 0x54, 0x5c
CYCLES_PER_PATCH = 4096   # what the fake kernel's counter says


def opt_header(name):
    """An optimized kernel's register map, x<name>_hw.h, as Vitis writes it."""
    regs = [('AP_CTRL', AP_CTRL)] + [(f'DIN{i}_DATA', a) for i, a in enumerate(DINS)] + \
        [('DOUT_DATA', OPT_DOUT), ('HEIGHT_DATA', HEIGHT), ('WIDTH_DATA', WIDTH), ('CYCLES_DATA', CYCLES)]
    return ''.join(f'#define X{name.upper()}_CONTROL_ADDR_{r} 0x{a:02x}\n' for r, a in regs)

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

typedef specpw_unet_qparams_golden<110, 3, specpw_unet_{size}> qparams_t;
static qparams_t params;
static int32_t din[32][32][110], dout[32][32][3];

int main(int, char** argv) {{
    SPECPW_UNET_LOAD_QPARAMS(params, sp_unet_{size}_int8);
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
    std::vector<uint16_t> cube((size_t)h * w * RAW_BANDS);
    FILE* f = fopen(argv[1], "rb");
    if (fread(cube.data(), 2, cube.size(), f) != cube.size()) return 1;
    fclose(f);
    std::vector<spu_raw_t> din(cube.size() / PORT_BANDS);
    for (size_t i = 0; i < cube.size(); i++) din[i / PORT_BANDS][i % PORT_BANDS] = cube[i];
    std::vector<spu_out_t> dout((size_t)h * w);
    uint32_t cycles = 0;
    {name}(din.data(), din.data(), din.data(), din.data(), dout.data(), h, w, cycles);
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
                xq = self.model.justounetsimple_model.preprocess_fixed(self.raw)
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
        real = m.justounetsimple_model.preprocess_fixed(self.patches(self.raw)).astype(np.int32)
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
        """The whole raw-cube kernels, Winograd (_opt) and direct (_opt_direct), on a real crop of
        45 x 70 pixels (2 x 3 patches, the edge ones partly outside): what classify_fixed and
        the class say, every value."""
        m = self.model
        cube = np.clip(np.rint(self.raw[:45, :70]), 0, 65535).astype(np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cube.tofile(tmp / 'cube.bin')
            for name, size in [(f'{m.name(size)}_opt{kind}', size) for size in m.SIZES for kind in ('', '_direct')]:
                with self.subTest(kernel=name):
                    source_dir = ROOT / 'src/hls' / name
                    compile_and_run(OPT_DRIVER.format(name=name), tmp, f'run_{name}',
                                    [source_dir, ROOT / 'src/common'], sorted(source_dir.glob('*.cpp')),
                                    [tmp / 'cube.bin', tmp / 'out.bin', 45, 70])
                    got = np.fromfile(tmp / 'out.bin', dtype=np.int32).reshape(45, 70, 4)
                    want = m.kernel_reference(cube, self.quantized(size))
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(len(np.unique(got[..., 3])), 3, 'all three classes occur')

    def test_fixed_writes_what_each_optimized_kernel_answers(self):
        m = self.model
        cube = np.clip(np.rint(self.raw[:40, :40]), 0, 65535).astype(np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)
            np.save(image / 'cube.npy', cube)
            written = m.fixed_reference(image, sizes=('small', 'large'))
            self.assertEqual([p.name for p in written],
                             ['reference_fixed_sp_unet_small.npy', 'reference_fixed_sp_unet_large.npy'])
            for size in ('small', 'large'):
                got = np.load(image / f'reference_fixed_sp_unet_{size}.npy')
                self.assertEqual(got.dtype, np.int32)
                np.testing.assert_array_equal(got[..., :3], m.classify_fixed(cube, self.quantized(size)))
                np.testing.assert_array_equal(got[..., 3], got[..., :3].argmax(-1))
            np.save(image / 'cube.npy', cube.astype(np.float32))
            with self.assertRaisesRegex(SystemExit, 'raw counts'):
                m.fixed_reference(image, sizes=('small',))

    def test_classify_fixed_cuts_the_image_like_the_notebook(self):
        """classify_fixed on a raw cube that is not whole patches: each patch's answer,
        the edge ones grown by repeating the last row and column."""
        m = self.model
        qlayers = self.quantized('small')
        cube = np.clip(np.rint(self.raw[:40, :50]), 0, 65535).astype(np.uint16)
        scores = m.classify_fixed(cube, qlayers)
        grown = np.pad(cube, ((0, 24), (0, 14), (0, 0)), mode='edge')
        prep = m.justounetsimple_model.prep_constants()
        for i in (0, 32):
            for j in (0, 32):
                want = m.forward_patch_fixed(m.justounetsimple_model.preprocess_fixed(
                    grown[i:i + PATCH, j:j + PATCH], prep), qlayers)
                np.testing.assert_array_equal(scores[i:i + PATCH, j:j + PATCH], want[:40 - i, :50 - j])


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass


class FakeKernel:
    """The HLS block's AXI-Lite registers; on start it computes n patches with the numpy
    model. Allocations larger than max_bytes fail, as when PYNQ's CMA is short."""

    def __init__(self, model, layers, broken=False, max_bytes=None):
        self.model, self.layers, self.broken, self.max_bytes = model, layers, broken, max_bytes
        self.words, self.buffers, self.done, self.runs = {}, {}, False, []
        self.next_address = 0x7000_0000

    def allocate(self, shape, dtype):
        if self.max_bytes is not None and np.prod(shape) * np.dtype(dtype).itemsize > self.max_bytes:
            raise RuntimeError('Failed to allocate Memory!')
        buffer = np.zeros(shape, dtype=dtype).view(FakeBuffer)
        buffer.physical_address = self.next_address
        self.buffers[self.next_address] = buffer
        self.next_address += 0x1_0000_0000   # the second buffer lands above 4 GiB: the high word matters
        return buffer

    def address(self, offset):
        return self.words.get(offset, 0) | (self.words.get(offset + 4, 0) << 32)

    def read(self, offset):
        if offset == AP_CTRL:
            done, self.done = self.done, False
            return (0x2 if done else 0) | 0x4
        return self.words.get(offset, 0)

    def write(self, offset, value):
        if offset == AP_CTRL and value & 1:
            n = self.words[N_REG]
            patches = np.asarray(self.buffers[self.address(DIN)]).reshape(-1, PATCH, PATCH, 110)
            scores = np.asarray(self.buffers[self.address(DOUT)]).reshape(-1, PATCH, PATCH, 3)
            for p in range(n):
                scores[p] = self.model.forward_patch(patches[p], self.layers)
            if self.broken:
                scores[n - 1, 5, 5, 0] += 0.5
            self.runs.append(n)
            self.done = True
        else:
            self.words[offset] = value


@unittest.skipUnless(np is not None, 'needs numpy')
class SpecPWUNetNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('specpw_unet_model', ROOT / 'tools/specpw_unet_model.py')
        if not cls.model.quantized_path('small').exists():
            raise unittest.SkipTest('no quantized weights: python tools/quantize_weights.py weights/fp32/sp_unet_small')
        cls.layers = cls.model.dequantize(cls.model.load_quantized('small'))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()        # a notebook's namespace (a cycle) may still map cube.npy; Windows locks it
        self.tmp.cleanup()

    def prepare(self, h, w, reference=True):
        """A board folder: the register map, mu_sd.txt and an image zip of h x w pixels. The
        raw values preprocess to N(0, 1); the dropped bands hold 1e6, so a wrong band
        selection blows the scores up."""
        kept, mean, inv_std = self.model.justounetsimple_model.read_preprocessing()
        z = np.random.default_rng(0).normal(0, 1, (h, w, 110)).astype(np.float32)
        cube = np.full((h, w, 120), 1.0e6, dtype=np.float32)
        cube[:, :, kept] = z / inv_std + mean
        image = self.dir / 'board' / 'aegean_unet'
        image.mkdir(parents=True)
        np.save(image / 'cube.npy', cube)
        np.save(image / 'labels.npy', (np.arange(h * w).reshape(h, w) % 4).astype(np.uint8))
        (image / 'meta.json').write_text(json.dumps({'rgb_bands': [56, 67, 86]}))
        if reference:
            self.model.reference(image, sizes=('small',))
        board = image.parent
        with zipfile.ZipFile(board / 'aegean_unet.zip', 'w') as z:
            for f in image.iterdir():
                z.write(f, f'aegean_unet/{f.name}')
        shutil.rmtree(image)
        (board / 'xsp_unet_small_hw.h').write_text(
            f'#define XSP_UNET_SMALL_CONTROL_ADDR_AP_CTRL   0x{AP_CTRL:02x}\n'
            f'#define XSP_UNET_SMALL_CONTROL_ADDR_DIN_DATA  0x{DIN:02x}\n'
            f'#define XSP_UNET_SMALL_CONTROL_ADDR_DOUT_DATA 0x{DOUT:02x}\n'
            f'#define XSP_UNET_SMALL_CONTROL_ADDR_N_DATA    0x{N_REG:02x}\n')
        shutil.copy2(ROOT / 'mu_sd.txt', board)
        return board

    def run_notebook(self, board, **fake):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib."""
        kernel = FakeKernel(self.model, self.layers, **fake)
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: types.SimpleNamespace(kernel=kernel)
        pynq.allocate = kernel.allocate
        pyplot = types.ModuleType('matplotlib.pyplot')
        class Anything:   # a figure or axes: every call does nothing
            def __getattr__(self, name):
                return lambda *a, **k: None
        pyplot.subplots = lambda n=1, m=1, **kw: (Anything(), [Anything() for _ in range(n * m)])
        pyplot.show = lambda *a, **k: None
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.pyplot = pyplot
        fakes = {'pynq': pynq, 'matplotlib': matplotlib, 'matplotlib.pyplot': pyplot}
        saved = {name: sys.modules.get(name) for name in fakes}
        sys.modules.update(fakes)
        cwd = Path.cwd()
        namespace = {}
        try:
            os.chdir(board)
            for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), namespace)
        finally:
            os.chdir(cwd)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return namespace, kernel

    def test_notebook_classifies_the_image_like_the_reference(self):
        board = self.prepare(50, 70)                 # not a multiple of 32: 2 x 3 patches
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ns, kernel = self.run_notebook(board)
        self.assertEqual(ns['NAME'], 'sp_unet_small')
        self.assertEqual(kernel.runs, [1, 6], 'the self-test patch, then all 6 in one start')
        self.assertEqual(ns['scores'].shape, (50, 70, 3))
        self.assertTrue(ns['close'].all())
        self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
        self.assertIn('PASS', printed.getvalue())
        self.assertIn('wall clock', printed.getvalue())
        self.assertTrue((board / 'aegean_unet_sp_unet_small_fpga_scores.npy').is_file())

    def test_notebook_goes_row_by_row_when_cma_is_short(self):
        board = self.prepare(50, 70)
        one_row = 3 * PATCH * PATCH * 110 * 4
        with contextlib.redirect_stdout(io.StringIO()):
            ns, kernel = self.run_notebook(board, max_bytes=one_row)
        self.assertEqual(kernel.runs, [1, 3, 3])
        self.assertTrue(ns['close'].all())

    def test_notebook_shows_a_wrong_kernel(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ns, _ = self.run_notebook(self.prepare(40, 40), broken=True)
        self.assertFalse(ns['close'].all())

    def test_notebook_without_a_reference_checks_the_labels_only(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ns, _ = self.run_notebook(self.prepare(40, 40, reference=False))
        self.assertIsNone(ns['reference'])
        self.assertIsNone(ns['close'])
        self.assertIn('no reference_sp_unet_small.npy', printed.getvalue())
        self.assertIn('accuracy against the labels', printed.getvalue())


class FakeImageKernel(FakeKernel):
    """An optimized kernel: a whole raw cube (height x width x 120 uint16) per start, at any
    address inside a buffer (strips start part-way into the scores), computed with the
    bit-exact numpy model."""

    def __init__(self, model, qlayers, broken=False, max_bytes=None):
        super().__init__(model, None, broken, max_bytes)
        self.qlayers, self.cache = qlayers, {}

    def view(self, address, shape, dtype):
        """shape x dtype at a physical address, inside one buffer."""
        for start, buffer in self.buffers.items():
            if start <= address and address + np.prod(shape) * np.dtype(dtype).itemsize <= start + buffer.nbytes:
                raw = np.asarray(buffer).reshape(-1).view(np.uint8)
                return raw[address - start:].view(dtype)[:np.prod(shape)].reshape(shape)
        raise AssertionError(f'0x{address:x}: no buffer holds {shape} {np.dtype(dtype)} there')

    def write(self, offset, value):
        if offset == AP_CTRL and value & 1:
            assert len({self.address(d) for d in DINS}) == 1, 'all input ports read the same cube'
            h, w = self.words[HEIGHT], self.words[WIDTH]
            cube = self.view(self.address(DINS[0]), (h, w, 120), np.uint16)
            key = cube.tobytes()
            if key not in self.cache:
                self.cache[key] = self.model.kernel_reference(cube, self.qlayers)
            out = self.cache[key].copy()
            if self.broken:
                out[min(5, h - 1), min(5, w - 1), 0] += 1 << 15
            self.view(self.address(OPT_DOUT), (h, w, 4), np.int32)[:] = out
            self.words[CYCLES] = CYCLES_PER_PATCH * (-(-h // 32) * -(-w // 32))
            self.runs.append((h, w))
            self.done = True
        else:
            self.words[offset] = value


@unittest.skipUnless(np is not None, 'needs numpy')
class SpecPWUNetOptNotebookTests(unittest.TestCase):
    """The optimized kernels' notebook (one for the three sizes) on a fake board."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('specpw_unet_model', ROOT / 'tools/specpw_unet_model.py')
        if not cls.model.quantized_path('medium').exists():
            raise unittest.SkipTest('no quantized weights: python tools/quantize_weights.py weights/fp32/sp_unet_medium')
        cls.qlayers = cls.model.load_quantized('medium')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()        # a notebook's namespace (a cycle) may still map cube.npy; Windows locks it
        self.tmp.cleanup()

    def prepare(self, h, w, references=True, name='sp_unet_medium_opt'):
        """A board folder: the register map and an image zip of h x w raw counts, whose
        preprocessing gives N(0, 1); the dropped bands hold 65535."""
        kept, mean, inv_std = self.model.justounetsimple_model.read_preprocessing()
        z = np.random.default_rng(0).normal(0, 1, (h, w, 110)).astype(np.float32)
        raw = np.full((h, w, 120), 65535.0, dtype=np.float32)
        raw[:, :, kept] = z / inv_std + mean
        image = self.dir / 'board' / 'aegean_unet'
        image.mkdir(parents=True)
        np.save(image / 'cube.npy', np.clip(np.rint(raw), 0, 65535).astype(np.uint16))
        np.save(image / 'labels.npy', (np.arange(h * w).reshape(h, w) % 4).astype(np.uint8))
        (image / 'meta.json').write_text(json.dumps({'rgb_bands': [56, 67, 86]}))
        if references:
            self.model.reference(image, sizes=('medium',))
            self.model.fixed_reference(image, sizes=('medium',))
        board = image.parent
        with zipfile.ZipFile(board / 'aegean_unet.zip', 'w') as z:
            for f in image.iterdir():
                z.write(f, f'aegean_unet/{f.name}')
        shutil.rmtree(image)
        (board / f'x{name}_hw.h').write_text(opt_header(name))
        return board

    def run_notebook(self, board, **fake):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib."""
        kernel = FakeImageKernel(self.model, self.qlayers, **fake)
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: types.SimpleNamespace(kernel=kernel)
        pynq.allocate = kernel.allocate
        pynq.Clocks = types.SimpleNamespace(fclk0_mhz=200.0)
        pyplot = types.ModuleType('matplotlib.pyplot')
        class Anything:   # a figure or axes: every call does nothing
            def __getattr__(self, name):
                return lambda *a, **k: None
        pyplot.subplots = lambda n=1, m=1, **kw: (Anything(), [Anything() for _ in range(n * m)])
        pyplot.show = lambda *a, **k: None
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.pyplot = pyplot
        fakes = {'pynq': pynq, 'matplotlib': matplotlib, 'matplotlib.pyplot': pyplot}
        saved = {name: sys.modules.get(name) for name in fakes}
        sys.modules.update(fakes)
        cwd = Path.cwd()
        namespace = {}
        try:
            os.chdir(board)
            for cell in json.loads(OPT_NOTEBOOK.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), namespace)
        finally:
            os.chdir(cwd)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return namespace, kernel

    def test_notebook_uses_the_kernel_formats(self):
        """The notebook and the numpy model hard-code what sp_unet_opt.hpp defines."""
        header = OPT_DESIGN.read_text(encoding='utf-8')

        def define(name):
            return int(re.search(rf'constexpr int {name} = (\d+);', header).group(1))

        setup = ''.join(json.loads(OPT_NOTEBOOK.read_text(encoding='utf-8'))['cells'][2]['source'])
        ns = {}
        exec(re.search(r'^BANDS, OUT_FRAC, LANES = .*$', setup, re.M).group(0), ns)
        self.assertEqual(ns['BANDS'], define('RAW_BANDS'))
        self.assertEqual(ns['OUT_FRAC'], define('OUT_FRAC'))
        self.assertEqual(ns['LANES'], 4, 'spu_out_t: 3 scores and the class')
        self.assertIn('hls::vector<int32_t, 4> spu_out_t', header)
        for name in ('IN_FRAC', 'ACT_FRAC', 'OUT_FRAC'):
            self.assertEqual(getattr(self.model, name), define(name), name)

    def test_notebook_classifies_the_whole_image_in_one_start(self):
        board = self.prepare(70, 50)                 # 3 x 2 patches, not a multiple of 32
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ns, kernel = self.run_notebook(board)
        self.assertEqual(ns['NAME'], 'sp_unet_medium_opt')
        self.assertEqual(ns['MODEL'], 'sp_unet_medium', 'the references of its network')
        self.assertEqual(ns['DINS'], list(DINS))
        # the self-test patch, the whole image, then 5 timed runs of the whole image
        self.assertEqual(kernel.runs, [(32, 32)] + [(70, 50)] * 6)
        self.assertIsNone(ns['strips'])
        self.assertEqual(ns['scores'].shape, (70, 50, 3))
        self.assertTrue(ns['close'].all())
        self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
        # against the float model (float z-score): the int8 input moves a few scores, hardly a class
        self.assertGreater((ns['classes'] == ns['reference'].argmax(-1)).mean(), 0.97)
        self.assertIn('PASS', printed.getvalue())
        self.assertIn('same class as the PC float model', printed.getvalue())
        self.assertAlmostEqual(ns['own'], CYCLES_PER_PATCH * 6 / 200e6)
        self.assertTrue((board / 'aegean_unet_sp_unet_medium_opt_fpga_scores.npy').is_file())

    def test_notebook_uses_strips_when_the_cube_does_not_fit(self):
        # 150 x 40 x 120 x 2 bytes = 1.44 MB does not fit, a 128-row strip (1.23 MB) does.
        board = self.prepare(150, 40)
        with contextlib.redirect_stdout(io.StringIO()):
            ns, kernel = self.run_notebook(board, max_bytes=1_300_000)
        self.assertIsNone(ns['cube_in'])
        self.assertEqual(kernel.runs[1:3], [(128, 40), (22, 40)], 'two strips, the second the rest')
        self.assertTrue(ns['close'].all())

    def test_notebook_shows_a_wrong_kernel(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ns, _ = self.run_notebook(self.prepare(40, 40), broken=True)
        self.assertFalse(ns['close'].all())

    def test_notebook_without_a_reference_checks_the_labels_only(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ns, _ = self.run_notebook(self.prepare(40, 40, references=False))
        self.assertIsNone(ns['reference_fixed'])
        self.assertIsNone(ns['close'])
        self.assertIn('no reference_fixed_sp_unet_medium.npy', printed.getvalue())
        self.assertIn('accuracy against the labels', printed.getvalue())


if __name__ == '__main__':
    unittest.main()
