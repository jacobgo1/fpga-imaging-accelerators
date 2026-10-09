"""justounetsimple: the numpy model against the C++ kernel, and the board notebook end to end.

The notebook's own code cells run against a fake `pynq` whose kernel computes each patch
with the numpy model, on an image folder made by tools/justounetsimple_image.py prepare.
That checks the register traffic, the patching and the comparison -- not PYNQ or the board.
"""
import gc
import importlib.util
import io
import json
from pathlib import Path
import re
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
NOTEBOOK = ROOT / 'software/pynq/justounetsimple/justounetsimple.ipynb'
OPT_NOTEBOOK = ROOT / 'software/pynq/justounetsimple_opt/justounetsimple_opt.ipynb'
CXX = shutil.which('g++') or shutil.which('clang++')
AP_CTRL, DIN, DOUT, N_REG = 0x00, 0x10, 0x1c, 0x28
HEADER = f"""\
#define XJUSTOUNETSIMPLE_CONTROL_ADDR_AP_CTRL   0x{AP_CTRL:02x}
#define XJUSTOUNETSIMPLE_CONTROL_ADDR_GIE       0x04
#define XJUSTOUNETSIMPLE_CONTROL_ADDR_DIN_DATA  0x{DIN:02x}
#define XJUSTOUNETSIMPLE_CONTROL_BITS_DIN_DATA  64
#define XJUSTOUNETSIMPLE_CONTROL_ADDR_DOUT_DATA 0x{DOUT:02x}
#define XJUSTOUNETSIMPLE_CONTROL_BITS_DOUT_DATA 64
"""
DINS, OPT_DOUT, HEIGHT, WIDTH = (0x10, 0x1c, 0x28, 0x34), 0x40, 0x4c, 0x54
DIN0 = DINS[0]
OPT_HEADER = f"""\
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_AP_CTRL     0x{AP_CTRL:02x}
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_DIN0_DATA   0x{DINS[0]:02x}
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_DIN1_DATA   0x{DINS[1]:02x}
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_DIN2_DATA   0x{DINS[2]:02x}
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_DIN3_DATA   0x{DINS[3]:02x}
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_DOUT_DATA   0x{OPT_DOUT:02x}
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_HEIGHT_DATA 0x{HEIGHT:02x}
#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_WIDTH_DATA  0x{WIDTH:02x}
"""
OPT_SOURCE = ROOT / 'src/hls/justounetsimple_opt'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass


class FakeKernel:
    """The HLS block's AXI-Lite registers; on start it computes the patch with the numpy model."""

    def __init__(self, model, layers, broken=False):
        self.model, self.layers, self.broken = model, layers, broken
        self.words, self.buffers, self.done, self.starts = {}, {}, False, 0
        self.next_address = 0x7000_0000

    def allocate(self, shape, dtype):
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
            self.starts += 1
            scores = self.model.forward_patch(np.array(self.buffers[self.address(DIN)]), self.layers)
            if self.broken:
                scores[5, 5, 0] += 0.5
            self.buffers[self.address(DOUT)][:] = scores
            self.done = True
        else:
            self.words[offset] = value


class FakeImageKernel(FakeKernel):
    """The optimized kernel: a whole raw cube (height x width x 120 uint16) per start, at any
    address inside a buffer (strips start part-way into the scores), computed with the
    bit-exact numpy model (classify_fixed). Allocations larger than max_bytes fail, as when
    PYNQ's CMA is short."""

    def __init__(self, model, layers, broken=False, max_bytes=None):
        super().__init__(model, layers, broken)
        self.qlayers, self.prep = model.load_quantized(), model.prep_constants()
        self.max_bytes, self.runs = max_bytes, []

    def allocate(self, shape, dtype):
        if self.max_bytes is not None and np.prod(shape) * np.dtype(dtype).itemsize > self.max_bytes:
            raise RuntimeError('Failed to allocate Memory!')
        return super().allocate(shape, dtype)

    def view(self, address, shape, dtype):
        """shape x dtype at a physical address, inside one buffer."""
        for start, buffer in self.buffers.items():
            if start <= address and address + np.prod(shape) * np.dtype(dtype).itemsize <= start + buffer.nbytes:
                raw = np.asarray(buffer).reshape(-1).view(np.uint8)
                return raw[address - start:].view(dtype)[:np.prod(shape)].reshape(shape)
        raise AssertionError(f'0x{address:x}: no buffer holds {shape} {np.dtype(dtype)} there')

    def write(self, offset, value):
        if offset == AP_CTRL and value & 1:
            self.starts += 1
            assert len({self.address(d) for d in DINS}) == 1, 'all input ports read the same cube'
            h, w = self.words[HEIGHT], self.words[WIDTH]
            cube = self.view(self.address(DIN0), (h, w, 120), np.uint16)
            scores = self.model.classify_fixed(cube, self.qlayers, self.prep)
            if self.broken:
                scores[min(5, h - 1), min(5, w - 1), 0] += 1 << 15
            self.view(self.address(OPT_DOUT), (h, w, 4), np.int32)[:] = self.model.with_class(scores)
            self.runs.append((h, w))
            self.done = True
        else:
            self.words[offset] = value


@unittest.skipIf(np is None, 'needs numpy')
class JustoUNetSimpleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('justounetsimple_model', ROOT / 'tools/justounetsimple_model.py')
        cls.patching = load_module('patching', ROOT / 'tools/patching.py')
        cls.tool = load_module('justounetsimple_image', ROOT / 'tools/justounetsimple_image.py')
        cls.layers = cls.model.load_weights()

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()        # a notebook's namespace (a cycle) may still map cube.npy; Windows locks it
        self.tmp.cleanup()

    def raw_cube(self, h, w, seed=0, counts=False):
        """Raw values whose preprocessing gives N(0, 1); the dropped bands hold 1e6, so a
        wrong band selection blows the scores up. counts: whole numbers 0 .. 65535 like a
        capture's, the dropped bands 65535."""
        kept, mean, inv_std = self.model.read_preprocessing()
        z = np.random.default_rng(seed).normal(0, 1, (h, w, 110)).astype(np.float32)
        raw = np.full((h, w, 120), 65535.0 if counts else 1.0e6, dtype=np.float32)
        raw[:, :, kept] = z / inv_std + mean
        return np.clip(np.rint(raw), 0, 65535) if counts else raw

    def prepare(self, h, w, header=('xjustounetsimple_hw.h', HEADER), counts=False):
        np.save(self.dir / 'source.npy', self.raw_cube(h, w, counts=counts))
        np.save(self.dir / 'source_labels.npy', (np.arange(h * w).reshape(h, w) % 4).astype(np.uint8))
        board = self.dir / 'board'
        board.mkdir()
        saved, sys.argv = sys.argv, ['justounetsimple_image.py', 'prepare', str(self.dir / 'source.npy'),
                                     '--labels', str(self.dir / 'source_labels.npy'),
                                     '--out', str(board / 'aegean_unet')]
        try:
            self.assertEqual(self.tool.main(), 0)
        finally:
            sys.argv = saved
        with zipfile.ZipFile(board / 'aegean_unet.zip', 'w') as z:
            for f in (board / 'aegean_unet').iterdir():
                z.write(f, f'aegean_unet/{f.name}')
        shutil.rmtree(board / 'aegean_unet')
        (board / header[0]).write_text(header[1])
        shutil.copy2(ROOT / 'mu_sd.txt', board)
        return board

    def run_notebook(self, board, broken=False, notebook=NOTEBOOK, kernel_class=FakeKernel):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib."""
        kernel = kernel_class(self.model, self.layers, broken)
        overlay = types.SimpleNamespace(kernel=kernel)
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: overlay
        pynq.allocate = kernel.allocate
        pyplot = types.ModuleType('matplotlib.pyplot')
        pyplot.subplots = lambda n, m, **kw: (None, [types.SimpleNamespace(
            imshow=lambda *a, **k: None, set_title=lambda *a: None, axis=lambda *a: None) for _ in range(m)])
        pyplot.show = lambda: None
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.pyplot = pyplot
        fakes = {'pynq': pynq, 'matplotlib': matplotlib, 'matplotlib.pyplot': pyplot}
        saved = {name: sys.modules.get(name) for name in fakes}
        sys.modules.update(fakes)
        cwd = Path.cwd()
        namespace = {}
        try:
            import os
            os.chdir(board)
            cells = json.loads(Path(notebook).read_text(encoding='utf-8'))['cells']
            for cell in cells:
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

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_numpy_model_matches_the_cpp_kernel(self):
        source = self.dir / 'run_kernel.cpp'
        source.write_text("""
#include "justounetsimple.hpp"
#include <cstdio>
static float din[JUNETS_H][JUNETS_W][JUNETS_IN_CH];
static float dout[JUNETS_H][JUNETS_W][JUNETS_OUT_CH];
int main(int, char** argv) {
    FILE* f = fopen(argv[1], "rb"); if (fread(din, sizeof din, 1, f) != 1) return 1; fclose(f);
    justounetsimple(din, dout);
    f = fopen(argv[2], "wb"); fwrite(dout, sizeof dout, 1, f); fclose(f);
}
""")
        exe = self.dir / 'run_kernel.exe'
        subprocess.run([CXX, '-std=c++14', '-O2', f'-I{ROOT / "src/hls/justounetsimple"}',
                        str(source), str(ROOT / 'src/hls/justounetsimple/justounetsimple.cpp'),
                        '-o', str(exe)], check=True)
        x = np.random.default_rng(3).uniform(-3, 3, (32, 32, 110)).astype(np.float32)
        x.tofile(self.dir / 'in.bin')
        subprocess.run([str(exe), str(self.dir / 'in.bin'), str(self.dir / 'out.bin')], check=True)
        cpp = np.fromfile(self.dir / 'out.bin', dtype=np.float32).reshape(32, 32, 3)
        want = self.model.forward_patch(x, self.layers)
        np.testing.assert_allclose(cpp, want, rtol=1e-3, atol=1e-3)
        self.assertGreater(float(want.max() - want.min()), 1.0, 'the reference should not be constant')

    def test_notebook_classifies_the_image_like_the_reference(self):
        board = self.prepare(50, 70)                 # not a multiple of 32: 2 x 3 patches
        ns, kernel = self.run_notebook(board)
        self.assertEqual(kernel.starts, 1 + 6, 'the self-test patch, then 6 patches')
        self.assertEqual(kernel.address(DIN), ns['patch_in'].physical_address)
        self.assertEqual(kernel.address(DOUT), ns['patch_out'].physical_address)
        self.assertEqual(ns['scores'].shape, (50, 70, 3))
        self.assertTrue(ns['close'].all())
        self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
        self.assertTrue((board / 'aegean_unet/fpga_scores.npy').is_file())

    def test_notebook_shows_a_wrong_kernel(self):
        ns, _ = self.run_notebook(self.prepare(40, 40), broken=True)
        self.assertFalse(ns['close'].all())

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_fixed_point_model_matches_the_optimized_cpp_kernel(self):
        """classify_fixed predicts the board's answers: it must be the optimized kernel's
        arithmetic bit for bit, the z-score and the edge patches included."""
        source = self.dir / 'run_opt.cpp'
        source.write_text("""
#include "justounetsimple_opt.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    std::vector<jopt_raw_t> din(h * w * JOPT_RAW_BANDS / JOPT_PORT_BANDS);
    std::vector<jopt_out_t> dout(h * w);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(jopt_raw_t), din.size(), f) != din.size()) return 1;
    fclose(f);
    const jopt_raw_t *d = din.data();
    justounetsimple_opt(d, d, d, d, dout.data(), h, w);
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(jopt_out_t), dout.size(), f); fclose(f);
}
""")
        exe = self.dir / 'run_opt.exe'
        subprocess.run([CXX, '-std=c++14', '-O2', f'-I{OPT_SOURCE}', f'-I{ROOT / "src/common"}',
                        str(source), str(OPT_SOURCE / 'justounetsimple_opt.cpp'), '-o', str(exe)], check=True)
        h, w = 37, 45                                     # 2 x 2 patches, the edge ones partial
        cube = self.raw_cube(h, w, seed=3, counts=True).astype(np.uint16)
        cube.tofile(self.dir / 'in.bin')
        subprocess.run([str(exe), str(self.dir / 'in.bin'), str(self.dir / 'out.bin'), str(h), str(w)],
                       check=True)
        cpp = np.fromfile(self.dir / 'out.bin', dtype=np.int32).reshape(h, w, 4)
        want = self.model.classify_fixed(cube)
        np.testing.assert_array_equal(cpp, self.model.with_class(want))
        # The float model on the same int8 input: only the rounding between the layers differs.
        scores = want * 2.0 ** -self.model.OUT_FRAC
        floats = np.zeros((h, w, 3), dtype=np.float32)
        for patch, tile in self.patching.iter_patches(cube, pad_mode='edge'):
            x = self.model.preprocess_fixed(patch) * np.float32(2.0 ** -self.model.IN_FRAC)
            self.patching.place(floats, self.model.forward_patch(x, self.layers), tile)
        np.testing.assert_allclose(scores, floats, atol=0.02)
        self.assertGreater(float(scores.max() - scores.min()), 1.0, 'the reference should not be constant')

    def test_with_class_picks_the_first_highest_score(self):
        scores = np.array([[[5, 7, 7], [9, 1, 9], [-3, -2, -4]]], dtype=np.int32)
        np.testing.assert_array_equal(self.model.with_class(scores)[..., 3], [[1, 0, 1]])
        np.testing.assert_array_equal(self.model.with_class(scores)[..., :3], scores)

    def test_kernel_input_is_int8_and_never_minus_128(self):
        """conv1 packs two products per DSP, which needs |input| <= 127."""
        x = self.model.quantize_input(np.array([-100.0, -3.99, 0.0, 3.99, 100.0]))
        self.assertEqual(x.dtype, np.int8)
        np.testing.assert_array_equal(x, [-127, -127, 0, 127, 127])
        raw = np.zeros((1, 1, 120), dtype=np.uint16)
        raw[..., 8:118] = 65535
        self.assertEqual(int(self.model.preprocess_fixed(raw).max()), 127)

    def test_integer_z_score_is_within_one_unit_of_the_float_one(self):
        cube = self.raw_cube(16, 16, counts=True)
        kept, mean, inv_std = self.model.read_preprocessing()
        floats = self.model.quantize_input(self.model.preprocess(cube, kept, mean, inv_std))
        ints = self.model.preprocess_fixed(cube.astype(np.uint16))
        self.assertLessEqual(int(np.abs(ints.astype(int) - floats).max()), 1)

    def test_prep_header_is_up_to_date(self):
        """justounetsimple_prep.hpp is generated from mu_sd.txt: regenerate it when that changes."""
        self.assertEqual(self.model.PREP_HEADER.read_text(encoding='utf-8'), self.model.prep_header(),
                         'run: python tools/justounetsimple_model.py')

    def test_opt_notebook_uses_the_kernel_formats(self):
        """The notebook and the numpy model hard-code what justounetsimple_opt.hpp defines."""
        header = (OPT_SOURCE / 'justounetsimple_opt.hpp').read_text(encoding='utf-8')

        def define(name):
            return int(re.search(rf'#define JOPT_{name}\s+(\d+)', header).group(1))

        setup = ''.join(json.loads(OPT_NOTEBOOK.read_text(encoding='utf-8'))['cells'][2]['source'])
        ns = {}
        exec(re.search(r'^BANDS, OUT_FRAC, LANES = .*$', setup, re.M).group(0), ns)
        self.assertEqual(ns['BANDS'], define('RAW_BANDS'))
        self.assertEqual(ns['OUT_FRAC'], define('OUT_FRAC'))
        self.assertEqual(ns['LANES'], 4, 'jopt_out_t: 3 scores and the class')
        self.assertIn('hls::vector<int32_t, 4> jopt_out_t', header)
        for name in ('IN_FRAC', 'ACT_FRAC', 'OUT_FRAC'):
            self.assertEqual(getattr(self.model, name), define(name), name)

    def run_opt(self, h, w, broken=False, max_bytes=None):
        board = self.prepare(h, w, ('xjustounetsimple_opt_hw.h', OPT_HEADER), counts=True)
        self.assertEqual(np.load(board.parent / 'source.npy').dtype, np.float32)
        kernel = lambda model, layers, broken: FakeImageKernel(model, layers, broken, max_bytes)
        return board, self.run_notebook(board, broken, OPT_NOTEBOOK, kernel)

    def test_opt_notebook_classifies_the_whole_image_in_one_start(self):
        board, (ns, kernel) = self.run_opt(70, 50)    # 3 x 2 patches, not a multiple of 32
        self.assertEqual(np.load(board / 'aegean_unet/cube.npy').dtype, np.uint16, 'saved as raw counts')
        self.assertEqual(ns['scores'].shape, (70, 50, 3))
        self.assertTrue(ns['close'].all())
        self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
        np.testing.assert_array_equal(ns['classes'], ns['raw_scores'][..., :3].argmax(-1))
        self.assertTrue((board / 'aegean_unet/reference_fixed.npy').is_file())
        self.assertIsNone(ns['strips'])
        # the self-test patch, the whole image, then 5 timed runs of the whole image
        self.assertEqual(kernel.runs, [(32, 32)] + [(70, 50)] * 6)
        for din in DINS:
            self.assertEqual(kernel.address(din), ns['cube_in'].physical_address)
        self.assertEqual(kernel.address(OPT_DOUT), ns['scores_out'].physical_address)
        self.assertGreater(ns['wall'], 0)
        self.assertGreater(ns['fpga'], 0)

    def test_opt_notebook_uses_strips_when_the_cube_does_not_fit(self):
        # 150 x 40 x 120 x 2 bytes = 1.44 MB does not fit, a 128-row strip (1.23 MB) does.
        board, (ns, kernel) = self.run_opt(150, 40, max_bytes=1_300_000)
        self.assertIsNone(ns['cube_in'])
        self.assertEqual(len(ns['strips']), 2)
        self.assertEqual(ns['scores'].shape, (150, 40, 3))
        self.assertTrue(ns['close'].all())
        self.assertEqual(kernel.runs[1:3], [(128, 40), (22, 40)], 'two strips, the second the rest')

    def test_opt_notebook_shows_a_wrong_kernel(self):
        _, (ns, _) = self.run_opt(40, 40, broken=True)
        self.assertFalse(ns['close'].all())

    def test_fixed_adds_the_kernel_reference_to_an_old_folder_or_zip(self):
        cube = self.raw_cube(40, 35, counts=True).astype(np.uint16)
        want = self.model.with_class(self.model.classify_fixed(cube))
        old = self.dir / 'old'
        old.mkdir()
        np.save(old / 'cube.npy', cube)
        with zipfile.ZipFile(self.dir / 'old.zip', 'w') as z:
            z.write(old / 'cube.npy', 'old/cube.npy')

        def fixed(image):
            saved, sys.argv = sys.argv, ['justounetsimple_image.py', 'fixed', str(image)]
            try:
                self.assertEqual(self.tool.main(), 0)
            finally:
                sys.argv = saved

        fixed(old)
        np.testing.assert_array_equal(np.load(old / 'reference_fixed.npy'), want)
        fixed(self.dir / 'old.zip')
        with zipfile.ZipFile(self.dir / 'old.zip') as z:
            self.assertEqual(sorted(z.namelist()), ['old/cube.npy', 'old/reference_fixed.npy'])
            np.testing.assert_array_equal(np.load(io.BytesIO(z.read('old/reference_fixed.npy'))), want)
        with self.assertRaises(SystemExit):
            fixed(self.dir / 'old.zip')          # already there: not added twice

    def test_notebook_padding_matches_the_reference_tiling(self):
        cube = self.raw_cube(37, 45)
        _, tiles = self.patching.cut(cube, pad_mode='edge')
        padded = self.patching.padded(cube, pad_mode='edge')
        self.assertEqual(padded.shape[:2], (64, 64))
        self.assertEqual(tiles, [(0, 0), (0, 32), (32, 0), (32, 32)])
        np.testing.assert_array_equal(padded[40, 50], cube[36, 44], 'edge pixels repeated')


if __name__ == '__main__':
    unittest.main()
