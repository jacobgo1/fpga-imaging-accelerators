"""justounetsimple_opt_readonly: its notebook, run against the C++ kernel itself.

The notebook's cells run here with a fake pynq whose kernel, on start, hands the cube to
the real kernel compiled with plain g++. So the notebook's numpy checksum is checked
against what the kernel computes, edge padding included, and the notebook's register
writes against the register map.
"""
import gc
import json
import os
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
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parents[1]
NAME = 'justounetsimple_opt_readonly'
NOTEBOOK = ROOT / 'software/pynq' / NAME / f'{NAME}.ipynb'
SOURCE = ROOT / 'src/hls' / NAME
CXX = shutil.which('g++') or shutil.which('clang++')
AP_CTRL, DINS, DOUT, HEIGHT, WIDTH, CYCLES = 0x00, (0x10, 0x1c, 0x28, 0x34), 0x40, 0x4c, 0x54, 0x5c
CYCLES_PER_PATCH = 3900   # what the fake kernel's counter says
HEADER = ''.join(f'#define XJUSTOUNETSIMPLE_OPT_READONLY_CONTROL_ADDR_{name} 0x{offset:02x}\n'
                 for name, offset in [('AP_CTRL', AP_CTRL)] + [(f'DIN{i}_DATA', d) for i, d in enumerate(DINS)]
                 + [('DOUT_DATA', DOUT), ('HEIGHT_DATA', HEIGHT), ('WIDTH_DATA', WIDTH),
                    ('CYCLES_DATA', CYCLES), ('CYCLES_CTRL', CYCLES + 4)])
DRIVER = f"""
#include "{NAME}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    std::vector<cube_word_t> din(h * w * CUBE_BANDS / BANDS_PER_WORD);
    std::vector<score_word_t> dout(1);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(cube_word_t), din.size(), f) != din.size()) return 1;
    fclose(f);
    cube_word_t *d = din.data();   // hls::burst_maxi takes a plain pointer
    uint32_t cycles;
    {NAME}(d, d, d, d, dout.data(), h, w, cycles);
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(score_word_t), 1, f); fclose(f);
}}
"""


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass


class FakeKernel:
    """The HLS block's registers; on start, the compiled C++ kernel on the cube at din0."""

    def __init__(self, exe, work, broken=False, max_bytes=None):
        self.exe, self.work, self.broken, self.max_bytes = exe, work, broken, max_bytes
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
        if offset != AP_CTRL or not value & 1:
            self.words[offset] = value
            return
        assert len({self.address(d) for d in DINS}) == 1, 'all input ports read the same cube'
        h, w = self.words[HEIGHT], self.words[WIDTH]
        cube = self.buffers[self.address(DINS[0])]
        assert cube.shape == (h, w, 120) and cube.dtype == np.uint16
        np.ascontiguousarray(cube).tofile(self.work / 'in.bin')
        subprocess.run([str(self.exe), str(self.work / 'in.bin'), str(self.work / 'out.bin'), str(h), str(w)],
                       check=True)
        result = np.fromfile(self.work / 'out.bin', dtype=np.int32)
        if self.broken:
            result[0] ^= 1
        self.buffers[self.address(DOUT)][:] = result
        self.words[CYCLES] = CYCLES_PER_PATCH * (-(-h // 32) * -(-w // 32))
        self.runs.append((h, w))
        self.done = True


@unittest.skipIf(np is None, 'needs numpy')
@unittest.skipUnless(CXX, 'no C++ compiler')
class ReadOnlyNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.build = Path(cls.tmp.name)
        (cls.build / 'driver.cpp').write_text(DRIVER)
        cls.exe = cls.build / 'kernel.exe'
        subprocess.run([CXX, '-std=c++14', '-O2', f'-I{SOURCE}', f'-I{ROOT / "src/common"}',
                        str(cls.build / 'driver.cpp'), str(SOURCE / f'{NAME}.cpp'), '-o', str(cls.exe)],
                       check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.tmp_board = tempfile.TemporaryDirectory()
        self.board = Path(self.tmp_board.name)

    def tearDown(self):
        gc.collect()        # the notebook's namespace may still map cube.npy; Windows locks it
        self.tmp_board.cleanup()

    def run_notebook(self, h, w, broken=False, max_bytes=None):
        cube = np.random.default_rng(h * 1000 + w).integers(0, 65536, (h, w, 120), dtype=np.uint16)
        (self.board / 'aegean_unet').mkdir()
        np.save(self.board / 'aegean_unet/cube.npy', cube)
        with zipfile.ZipFile(self.board / 'aegean_unet.zip', 'w') as z:
            z.write(self.board / 'aegean_unet/cube.npy', 'aegean_unet/cube.npy')
        shutil.rmtree(self.board / 'aegean_unet')
        (self.board / f'x{NAME}_hw.h').write_text(HEADER)

        kernel = FakeKernel(self.exe, self.board, broken, max_bytes)
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: types.SimpleNamespace(kernel=kernel)
        pynq.allocate = kernel.allocate
        pynq.Clocks = types.SimpleNamespace(fclk0_mhz=200.0)
        saved = sys.modules.get('pynq')
        sys.modules['pynq'] = pynq
        cwd, namespace = Path.cwd(), {}
        try:
            os.chdir(self.board)
            for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), namespace)
        finally:
            os.chdir(cwd)
            if saved is None:
                sys.modules.pop('pynq', None)
            else:
                sys.modules['pynq'] = saved
        return cube, namespace, kernel

    def test_checksum_matches_the_kernel_on_a_whole_image(self):
        cube, ns, kernel = self.run_notebook(70, 45)       # 3 x 2 patches, the edge ones partial
        self.assertTrue(ns['ok'])
        self.assertEqual(ns['NAME'], NAME)
        self.assertEqual(ns['DINS'], list(DINS), 'the input ports, from the register map')
        self.assertEqual(kernel.runs, [(70, 45)] * 6, 'the check, then 5 timed runs')
        for din in DINS:
            self.assertEqual(kernel.address(din), ns['cube_in'].physical_address)
        self.assertEqual(kernel.address(DOUT), ns['out'].physical_address)
        self.assertEqual(ns['PATCHES'], 6)
        self.assertEqual(ns['READ_CYCLES'], 3840)
        self.assertGreater(ns['best'], 0)
        self.assertEqual(ns['CYCLES'], CYCLES, 'the counter, from the register map')
        self.assertAlmostEqual(ns['own_best'], CYCLES_PER_PATCH * 6 / 200e6)
        self.assertAlmostEqual(ns['cycles'], CYCLES_PER_PATCH)
        # Without the padding the sum would differ: the notebook must count it.
        plain = (cube.reshape(-1, 8).sum(0, dtype=np.uint64) % 65536).astype(np.uint16)
        self.assertFalse((ns['checksum'] == plain).all())

    def test_first_rows_when_the_cube_does_not_fit(self):
        # 150 x 40 x 120 x 2 bytes = 1.44 MB does not fit, 128 rows (1.23 MB) do.
        _, ns, kernel = self.run_notebook(150, 40, max_bytes=1_300_000)
        self.assertTrue(ns['ok'])
        self.assertEqual(kernel.runs[0], (128, 40))

    def test_notebook_shows_a_wrong_checksum(self):
        _, ns, _ = self.run_notebook(40, 40, broken=True)
        self.assertFalse(ns['ok'])


if __name__ == '__main__':
    unittest.main()
