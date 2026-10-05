"""justounetsimple: the numpy model against the C++ kernel, and the board notebook end to end.

The notebook's own code cells run against a fake `pynq` whose kernel computes each patch
with the numpy model, on an image folder made by tools/justounetsimple_image.py prepare.
That checks the register traffic, the patching and the comparison -- not PYNQ or the board.
"""
import importlib.util
import json
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
OPT_HEADER = (HEADER.replace('XJUSTOUNETSIMPLE_', 'XJUSTOUNETSIMPLE_OPT_')
              + f'#define XJUSTOUNETSIMPLE_OPT_CONTROL_ADDR_N_DATA    0x{N_REG:02x}\n')


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


class FakeBatchKernel(FakeKernel):
    """The optimized kernel: n patches per start, input as [55][32][32][2] per patch."""

    def write(self, offset, value):
        if offset == AP_CTRL and value & 1:
            n = self.words[N_REG]
            din, dout = self.buffers[self.address(DIN)], self.buffers[self.address(DOUT)]
            for k in range(n):
                self.starts += 1
                patch = np.array(din[k]).transpose(1, 2, 0, 3).reshape(32, 32, 110)
                scores = self.model.forward_patch(patch, self.layers)
                if self.broken:
                    scores[5, 5, 0] += 0.5
                dout[k] = scores
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
        self.tmp.cleanup()

    def raw_cube(self, h, w, seed=0):
        """Raw values whose preprocessing gives N(0, 1); the dropped bands hold 1e6, so a
        wrong band selection blows the scores up."""
        kept, mean, inv_std = self.model.read_preprocessing()
        z = np.random.default_rng(seed).normal(0, 1, (h, w, 110)).astype(np.float32)
        raw = np.full((h, w, 120), 1.0e6, dtype=np.float32)
        raw[:, :, kept] = z / inv_std + mean
        return raw

    def prepare(self, h, w, header=('xjustounetsimple_hw.h', HEADER)):
        np.save(self.dir / 'source.npy', self.raw_cube(h, w))
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

    def run_opt(self, h, w, broken=False):
        board = self.prepare(h, w, ('xjustounetsimple_opt_hw.h', OPT_HEADER))
        return board, self.run_notebook(board, broken, OPT_NOTEBOOK, FakeBatchKernel)

    def test_opt_notebook_classifies_the_image_like_the_reference(self):
        board, (ns, kernel) = self.run_opt(70, 50)    # 3 x 2 patches, not a multiple of 32
        self.assertEqual(ns['scores'].shape, (70, 50, 3))
        self.assertTrue(ns['close'].all())
        self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
        self.assertEqual(kernel.address(DIN), ns['patches_in'].physical_address)

    def test_opt_notebook_splits_large_images_into_batches(self):
        board, (ns, kernel) = self.run_opt(40, 40)
        ns['BATCH'] = 1                               # pretend the buffers hold one patch
        scores = ns['classify'](ns['cube'])
        np.testing.assert_allclose(scores, ns['reference'], rtol=1e-4, atol=1e-4)

    def test_opt_notebook_shows_a_wrong_kernel(self):
        _, (ns, _) = self.run_opt(40, 40, broken=True)
        self.assertFalse(ns['close'].all())

    def test_notebook_padding_matches_the_reference_tiling(self):
        cube = self.raw_cube(37, 45)
        _, tiles = self.patching.cut(cube, pad_mode='edge')
        padded = self.patching.padded(cube, pad_mode='edge')
        self.assertEqual(padded.shape[:2], (64, 64))
        self.assertEqual(tiles, [(0, 0), (0, 32), (32, 0), (32, 32)])
        np.testing.assert_array_equal(padded[40, 50], cube[36, 44], 'edge pixels repeated')


if __name__ == '__main__':
    unittest.main()
