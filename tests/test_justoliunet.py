"""justoliunet and justoliunet_bn: the numpy model against the trained model and the C++ kernels.

The numpy model (tools/justoliunet_model.py) is checked against the old PC reference of
the aegean pixels and against the labels; the C++ kernels, compiled with plain g++,
against the numpy model on the same real pixels. The kernels' testbenches compare them
with a float network built from the same golden modules (relu1d, maxpool1d, flatten);
this test is the independent check of those modules.

The board notebook's code cells run against a fake `pynq` whose kernel computes the
pixels with this numpy model: that checks the register traffic, the preprocessing,
the chunking and the comparison -- not PYNQ or the board.
"""
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tracemalloc
import types
import unittest
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which('g++') or shutil.which('clang++')
MODELS = ('justoliunet', 'justoliunet_bn')
NOTEBOOK = ROOT / 'software/pynq/justoliunet/justoliunet.ipynb'
AP_CTRL, DIN, DOUT, N_REG = 0x00, 0x10, 0x1c, 0x28


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantized(model):
    return ROOT / 'weights/quantized' / model / f'{model}_int8.npz'


def header(model):
    name = model.upper()
    return (f'#define X{name}_CONTROL_ADDR_AP_CTRL   0x{AP_CTRL:02x}\n'
            f'#define X{name}_CONTROL_ADDR_DIN_DATA  0x{DIN:02x}\n'
            f'#define X{name}_CONTROL_ADDR_DOUT_DATA 0x{DOUT:02x}\n'
            f'#define X{name}_CONTROL_ADDR_N_DATA    0x{N_REG:02x}\n')


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass


class FakeKernel:
    """The HLS block's AXI-Lite registers; on start it computes n pixels with the numpy model,
    at any address inside a buffer (strips write part-way into the scores). Allocations
    larger than max_bytes fail, as when PYNQ's CMA is short."""

    def __init__(self, model, weights, broken=False, max_bytes=None):
        self.model, self.weights, self.broken, self.max_bytes = model, weights, broken, max_bytes
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

    def view(self, address, shape):
        """shape float32 at a physical address, inside one buffer."""
        for start, buffer in self.buffers.items():
            if start <= address and address + np.prod(shape) * 4 <= start + buffer.nbytes:
                assert buffer.dtype == np.float32
                return np.asarray(buffer).reshape(-1)[(address - start) // 4:][:np.prod(shape)].reshape(shape)
        raise AssertionError(f'0x{address:x}: no buffer holds {shape} float32 there')

    def write(self, offset, value):
        if offset == AP_CTRL and value & 1:
            n = self.words[N_REG]
            spectra = self.view(self.address(DIN), (n, 110))
            scores = self.view(self.address(DOUT), (n, 3))
            for i in range(0, n, 1024):    # in pieces: on the board this memory is not the ARM's
                scores[i:i + 1024] = self.model.forward(spectra[i:i + 1024].astype(np.float64), self.weights)
            if self.broken:
                scores[min(5, n - 1), 0] += 0.5
            self.runs.append(n)
            self.done = True
        else:
            self.words[offset] = value


@unittest.skipUnless(np is not None, 'needs numpy')
class JustoLiuNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('justoliunet_model', ROOT / 'tools/justoliunet_model.py')
        kept, mean, inv_std = cls.model.read_preprocessing()
        raw, cls.labels = cls.model.aegean_pixels()
        cls.x = cls.model.preprocess(raw, kept, mean, inv_std)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def test_fp32_model_reproduces_the_old_reference(self):
        """valid convs, pools that drop an odd last position, channel-major flatten:
        the old justoliunet kernel's PC reference of the aegean pixels, to rounding."""
        reference = np.load(ROOT / 'aegean/reference_logits.npy')
        np.testing.assert_allclose(self.model.forward_fp32(self.x, 'justoliunet'), reference, atol=1e-9)

    def test_batchnorm_sits_between_conv1_and_its_relu(self):
        """conv1 -> bn1 -> ReLU is the trained order (folding bn1 into conv1 relies on it):
        it classifies the labelled pixels as well as training reported (test accuracy 93 %)."""
        scores = self.model.forward_fp32(self.x, 'justoliunet_bn')
        self.assertGreater((scores.argmax(1) == self.labels).mean(), 0.95)

    def test_int8_weights_keep_the_classes(self):
        for model in MODELS:
            with self.subTest(model=model):
                if not quantized(model).exists():
                    self.skipTest(f'no {quantized(model).relative_to(ROOT)}: '
                                  f'python tools/quantize_weights.py weights/fp32/{model}')
                fp32 = self.model.forward_fp32(self.x, model).argmax(1)
                int8 = self.model.forward(self.x, self.model.load_quantized(model)).argmax(1)
                self.assertGreater((int8 == fp32).mean(), 0.99)
                self.assertLess(abs((int8 == self.labels).mean() - (fp32 == self.labels).mean()), 0.01)

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_cpp_kernels_match_the_numpy_model_on_real_pixels(self):
        pixels = self.x[:2000].astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pixels.tofile(tmp / 'in.bin')
            for model in MODELS:
                with self.subTest(model=model):
                    if not quantized(model).exists():
                        self.skipTest(f'no quantized weights for {model}')
                    source_dir = ROOT / 'src/hls' / model
                    source = tmp / f'run_{model}.cpp'
                    source.write_text(f"""
#include "{model}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int n = atoi(argv[3]);
    std::vector<float> din(n * JLN_BANDS), dout(n * JLN_CLASSES);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(float), din.size(), f) != din.size()) return 1;
    fclose(f);
    {model}(din.data(), dout.data(), n);
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(float), dout.size(), f); fclose(f);
}}
""")
                    exe = tmp / f'run_{model}.exe'
                    subprocess.run([CXX, '-std=c++14', '-O2', f'-I{source_dir}', f'-I{ROOT / "src/common"}',
                                    str(source), str(source_dir / f'{model}.cpp'), '-o', str(exe)], check=True)
                    out = tmp / f'out_{model}.bin'
                    subprocess.run([str(exe), str(tmp / 'in.bin'), str(out), str(len(pixels))], check=True)
                    cpp = np.fromfile(out, dtype=np.float32).reshape(len(pixels), 3)
                    want = self.model.forward(pixels.astype(np.float64), self.model.load_quantized(model))
                    np.testing.assert_allclose(cpp, want, rtol=1e-4, atol=1e-4)
                    self.assertTrue((cpp.argmax(1) == want.argmax(1)).all())


@unittest.skipUnless(np is not None, 'needs numpy')
class JustoLiuNetNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('justoliunet_model', ROOT / 'tools/justoliunet_model.py')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()        # a notebook's namespace (a cycle) may still hold cube.npy; Windows locks it
        self.tmp.cleanup()

    def prepare(self, model, h, w):
        """The board's folder: what the kernel's package and an image zip put there. The
        image is aegean pixels (real spectra, so real classes) with the dropped bands at
        65535, so a wrong band selection blows the scores up."""
        if not quantized(model).exists():
            self.skipTest(f'no {quantized(model).relative_to(ROOT)}: '
                          f'python tools/quantize_weights.py weights/fp32/{model}')
        raw, labels = self.model.aegean_pixels()
        cube = np.rint(raw[:h * w]).astype(np.uint16).reshape(h, w, 120)
        kept, _, _ = self.model.read_preprocessing()
        cube[:, :, [b for b in range(120) if b not in kept]] = 65535
        board = self.dir / model
        image = board / 'aegean_unet'
        image.mkdir(parents=True)
        np.save(image / 'cube.npy', cube)
        np.save(image / 'labels.npy', labels[:h * w].reshape(h, w).astype(np.uint8))
        (image / 'meta.json').write_text(json.dumps({'rgb_bands': [56, 67, 86]}))
        with zipfile.ZipFile(board / 'aegean_unet.zip', 'w') as z:
            for f in image.iterdir():
                z.write(f, f'aegean_unet/{f.name}')
        shutil.rmtree(image)
        (board / f'x{model}_hw.h').write_text(header(model))
        shutil.copy2(ROOT / 'mu_sd.txt', board)
        shutil.copy2(quantized(model), board)
        return board

    def run_notebook(self, board, model, broken=False, max_bytes=None):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib."""
        kernel = FakeKernel(self.model, self.model.load_quantized(model), broken, max_bytes)
        overlay = types.SimpleNamespace(kernel=kernel)
        bitstreams = []
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: (bitstreams.append(path), overlay)[1]
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
        self.assertEqual(bitstreams, [f'{model}.bit'])
        return namespace, kernel

    def test_notebook_classifies_the_image_like_the_numpy_model(self):
        for model in MODELS:
            with self.subTest(model=model):
                ns, kernel = self.run_notebook(self.prepare(model, 50, 70), model)
                self.assertEqual(ns['NAME'], model)
                self.assertIsNone(ns['strips'])
                self.assertEqual(kernel.runs, [256] + [50 * 70] * 6,
                                 'the self-test, the whole image in one start, then 5 for the speed')
                self.assertEqual(kernel.address(DIN), ns['spectra'].physical_address)
                self.assertEqual(kernel.address(DOUT), ns['scores_out'].physical_address)
                self.assertEqual(ns['scores'].shape, (50, 70, 3))
                self.assertTrue(ns['close'].all())
                self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
                self.assertGreater((ns['classes'] == ns['labels']).mean(), 0.9)
                self.assertTrue((self.dir / model / 'aegean_unet_fpga_scores.npy').is_file())

    def test_notebook_uses_strips_when_the_image_does_not_fit(self):
        # 300 rows: strips of 128, 128 and 44 rows; the whole image's spectra do not fit in CMA.
        h, w = 300, 30
        ns, kernel = self.run_notebook(self.prepare('justoliunet', h, w), 'justoliunet',
                                       max_bytes=h * w * 110 * 4 - 1)
        self.assertIsNotNone(ns['strips'])
        self.assertEqual(kernel.runs, [256, 128 * w, 128 * w, 44 * w] + [128 * w] * 5)
        self.assertTrue(ns['close'].all())
        self.assertGreater((ns['classes'] == ns['labels']).mean(), 0.9)

    def test_notebook_checks_the_image_in_little_memory(self):
        # The reference needs about 13 KB a pixel; a 128-row strip of a real capture (1092 wide)
        # is 140 K pixels, 1.75 GB, more than the board has. 10 000 pixels in one strip here:
        # about 128 MB in one call, under 100 MB only if the reference is fed in pieces.
        board = self.prepare('justoliunet', 100, 100)
        tracemalloc.start()
        try:
            ns, _ = self.run_notebook(board, 'justoliunet')
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertTrue(ns['close'].all())
        self.assertLess(peak, 100 * 2**20, f'the notebook peaked at {peak / 2**20:.0f} MB')

    def test_notebook_shows_a_wrong_kernel(self):
        ns, _ = self.run_notebook(self.prepare('justoliunet', 40, 40), 'justoliunet', broken=True)
        self.assertFalse(ns['close'].all())


if __name__ == '__main__':
    unittest.main()
