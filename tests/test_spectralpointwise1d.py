"""spectralpointwise1d (golden) and spectralpointwise1d_opt with its Zynq-7030 and Zynq-7020
versions: the numpy models against the trained model, every C++ kernel against them on the
labelled aegean pixels (the optimized ones bit for bit, the same answers on every board), the
reference files for the OPU's classify, the packages, and both notebooks end to end against a
fake pynq whose kernel is the numpy model.

classify_fixed (tools/spectralpointwise1d_model.py) is what the optimized kernels must write,
bit for bit; here it is checked against the trained fp32 model on real pixels, and the
kernels, compiled with plain g++, against it. The kernels' testbenches check them against a
plain-loop integer reference of their own and the float golden network.
"""
import contextlib
import gc
import importlib.util
import io
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
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which('g++') or shutil.which('clang++')
MODEL = 'spectralpointwise1d'
# Every optimized kernel and its call on din, dout, height, width, cycles: every port reads the cube.
OPTIMIZED = {'spectralpointwise1d_opt': 'spectralpointwise1d_opt(d, d, d, d, dout.data(), h, w, cycles)',
             'spectralpointwise1d_opt_zynq7030': 'spectralpointwise1d_opt_zynq7030(d, d, d, d, dout.data(), h, w, cycles)',
             'spectralpointwise1d_opt_zynq7020': 'spectralpointwise1d_opt_zynq7020(d, d, dout.data(), h, w, cycles)'}
NOTEBOOK = ROOT / 'software/pynq/spectralpointwise1d/spectralpointwise1d.ipynb'
OPT_NOTEBOOK = ROOT / 'software/pynq/spectralpointwise1d_opt/spectralpointwise1d_opt.ipynb'
# Register offsets for the fake kernels, in the layout Vitis HLS gives them.
GOLDEN_REGS = {'AP_CTRL': 0x00, 'DIN_DATA': 0x10, 'DOUT_DATA': 0x1c, 'N_DATA': 0x28}
OPT_REGS = {'AP_CTRL': 0x00, 'DIN0_DATA': 0x10, 'DIN1_DATA': 0x1c, 'DIN2_DATA': 0x28, 'DIN3_DATA': 0x34,
            'DOUT_DATA': 0x40, 'HEIGHT_DATA': 0x4c, 'WIDTH_DATA': 0x54, 'CYCLES_DATA': 0x5c}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_and_run(tmp, kernel, body, inputs, out_dtype, out_shape):
    """Compile src/hls/KERNEL/ with a main() around body; run it on inputs; its output."""
    source_dir = ROOT / 'src/hls' / kernel
    source = tmp / f'run_{kernel}.cpp'
    source.write_text(body)
    exe = tmp / f'run_{kernel}.exe'
    subprocess.run([CXX, '-std=c++14', '-O2', f'-I{source_dir}', f'-I{ROOT / "src/common"}',
                    str(source), str(source_dir / f'{kernel}.cpp'), '-o', str(exe)], check=True)
    out = tmp / f'out_{kernel}.bin'
    subprocess.run([str(exe), str(inputs), str(out), *[str(v) for v in out_shape[:-1]]], check=True)
    return np.fromfile(out, dtype=out_dtype).reshape(out_shape)


@unittest.skipUnless(np is not None, 'needs numpy')
class SpectralPointwise1DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('spectralpointwise1d_model', ROOT / 'tools/spectralpointwise1d_model.py')
        cls.raw, cls.labels = cls.model.aegean_pixels()
        kept, mean, inv_std = cls.model.read_preprocessing()
        cls.x = cls.model.preprocess(cls.raw, kept, mean, inv_std)
        if not cls.model.quantized_path().exists():
            raise unittest.SkipTest(f'no quantized weights: python tools/quantize_weights.py weights/fp32/{MODEL}')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def test_batchnorm_sits_between_convsp1_and_its_relu(self):
        """convsp1 -> bnsp1 -> ReLU is the trained order (folding it into convsp1 relies on it):
        it classifies the labelled pixels as well as training reported (test accuracy 94 %)."""
        scores = self.model.forward_fp32(self.x)
        self.assertGreater((scores.argmax(1) == self.labels).mean(), 0.95)

    def test_int8_weights_keep_the_classes(self):
        fp32 = self.model.forward_fp32(self.x).argmax(1)
        int8 = self.model.forward(self.x, self.model.load_quantized()).argmax(1)
        self.assertGreater((int8 == fp32).mean(), 0.99)
        self.assertLess(abs((int8 == self.labels).mean() - (fp32 == self.labels).mean()), 0.01)

    def test_formats_header_is_up_to_date(self):
        """src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt_formats.hpp comes from the model's formats."""
        self.assertEqual(self.model.FORMATS_HEADER.read_text(), self.model.formats_header(),
                         'run: python tools/spectralpointwise1d_model.py formats')

    def test_integer_model_keeps_the_trained_classes(self):
        """12-bit input, uint8 activations: as good as int8 weights with float activations."""
        fp32 = self.model.forward_fp32(self.x).argmax(1)
        fixed = self.model.classify_fixed(self.raw)
        self.assertGreater((fixed[:, 3] == fp32).mean(), 0.995)
        self.assertLess(abs((fixed[:, 3] == self.labels).mean() - (fp32 == self.labels).mean()), 0.01)
        # the class lane is the first highest score
        np.testing.assert_array_equal(fixed[:, 3], fixed[:, :3].argmax(1))

    def test_activations_fit_uint8_on_real_pixels(self):
        """ACT_FRAC 4: up to 15.9, above the largest activation of the aegean pixels (about
        11), so on real pixels nothing saturates."""
        w = self.model.load_quantized()
        h1 = np.maximum(self.x @ w['convsp1.weight'].T + w['convsp1.bias'], 0)
        h2 = np.maximum(h1 @ w['convsp2.weight'].T + w['convsp2.bias'], 0)
        for h, frac in zip((h1, h2), self.model.ACT_FRACS):
            self.assertLess(h.max(), self.model.ACT_MAX * 2.0 ** -frac)

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_cpp_golden_kernel_matches_the_numpy_model_on_real_pixels(self):
        pixels = self.x[:2000].astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pixels.tofile(tmp / 'in.bin')
            cpp = compile_and_run(tmp, MODEL, f"""
#include "{MODEL}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int n = atoi(argv[3]);
    std::vector<float> din(n * SPW_BANDS), dout(n * SPW_CLASSES);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(float), din.size(), f) != din.size()) return 1;
    fclose(f);
    {MODEL}(din.data(), dout.data(), n);
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(float), dout.size(), f); fclose(f);
}}
""", tmp / 'in.bin', np.float32, (len(pixels), 3))
        want = self.model.forward(pixels.astype(np.float64), self.model.load_quantized())
        np.testing.assert_allclose(cpp, want, rtol=1e-4, atol=1e-4)
        self.assertTrue((cpp.argmax(1) == want.argmax(1)).all())

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_cpp_optimized_kernels_are_the_integer_model_bit_for_bit(self):
        """Every board's kernel, on the 10 275 aegean pixels as 3 x 3425 (the lanes' blocks
        run across rows) and as one row of the first 1001 (a short last block)."""
        want = self.model.classify_fixed(self.raw)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.raw.astype(np.uint16).tofile(tmp / 'in.bin')
            for kernel, call in OPTIMIZED.items():
                for h, w in ((3, 3425), (1, 1001)):
                    with self.subTest(kernel=kernel, image=(h, w)):
                        cpp = compile_and_run(tmp, kernel, f"""
#include "{kernel}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    std::vector<cube_word_t> din(h * w * WORDS_PER_PIXEL);
    std::vector<score_word_t> dout(h * w);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(cube_word_t), din.size(), f) != din.size()) return 1;
    fclose(f);
    uint32_t cycles = 0;
    cube_word_t *d = din.data();     // hls::burst_maxi takes a plain pointer
    {call};
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(score_word_t), dout.size(), f); fclose(f);
}}
""", tmp / 'in.bin', np.int32, (h, w, 4))
                        np.testing.assert_array_equal(cpp.reshape(-1, 4), want[:h * w])

    def test_reference_file_is_the_integer_model(self):
        """tools/spectralpointwise1d_model.py reference IMAGE: what the optimized kernels must
        write for IMAGE/cube.npy, in chunks that need not divide the image."""
        h, w = 23, 41
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / 'cube.npy', self.raw[:h * w].astype(np.uint16).reshape(h, w, 120))
            path = self.model.reference(tmp, chunk=100)
            self.assertEqual(path.name, 'reference_spectralpointwise1d.npy')
            got = np.load(path)
            self.assertEqual((got.shape, got.dtype), ((h, w, 4), np.int32))
            np.testing.assert_array_equal(got.reshape(-1, 4), self.model.classify_fixed(self.raw[:h * w]))

    def test_petalinux_kernels_have_the_opu_program(self):
        """A zedboard or zynq7030 build packs software/opu/KERNEL/ (main.py): the generic classify.c."""
        for kernel in ('spectralpointwise1d_opt_zynq7020', 'spectralpointwise1d_opt_zynq7030'):
            with self.subTest(kernel=kernel):
                include = ROOT / 'software/opu' / kernel / 'include.txt'
                files = [line.split('#')[0].strip() for line in include.read_text().splitlines()]
                files = [f for f in files if f]
                self.assertEqual(files, ['software/opu/justounetsimple_opt/classify.c'])
                self.assertTrue((ROOT / files[0]).is_file())

    def test_pynq_packages_name_existing_files(self):
        for kernel in ('spectralpointwise1d', 'spectralpointwise1d_opt', 'spectralpointwise1d_opt_zynq7020'):
            with self.subTest(kernel=kernel):
                include = ROOT / 'software/pynq' / kernel / 'include.txt'
                files = [line.split('#')[0].strip() for line in include.read_text().splitlines()]
                for f in filter(None, files):
                    self.assertTrue((ROOT / f).is_file(), f)


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass


class FakeKernel:
    """The HLS block's AXI-Lite registers. On start, the golden kernel (regs GOLDEN_REGS)
    computes n spectra with the numpy model; the optimized one (OPT_REGS) classifies the
    height x width cube at din0 with the integer model. Allocations larger than max_bytes
    fail, as when PYNQ's CMA is short."""

    def __init__(self, tools, regs, broken=False, max_bytes=None):
        self.tools, self.regs, self.broken, self.max_bytes = tools, regs, broken, max_bytes
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

    def address(self, name):
        offset = self.regs[name]
        return self.words.get(offset, 0) | (self.words.get(offset + 4, 0) << 32)

    def view(self, address, count, dtype):
        for start, buffer in self.buffers.items():
            if start <= address and address + count * np.dtype(dtype).itemsize <= start + buffer.nbytes:
                assert buffer.dtype == dtype
                return np.asarray(buffer).reshape(-1)[(address - start) // np.dtype(dtype).itemsize:][:count]
        raise AssertionError(f'0x{address:x}: no buffer holds {count} {np.dtype(dtype)} there')

    def read(self, offset):
        if offset == self.regs['AP_CTRL']:
            done, self.done = self.done, False
            return (0x2 if done else 0) | 0x4
        return self.words.get(offset, 0)

    def write(self, offset, value):
        if offset != self.regs['AP_CTRL'] or not value & 1:
            self.words[offset] = value
            return
        if 'N_DATA' in self.regs:          # the golden kernel: preprocessed spectra in, float scores out
            n = self.words[self.regs['N_DATA']]
            spectra = self.view(self.address('DIN_DATA'), n * 110, np.float32).reshape(n, 110)
            scores = self.view(self.address('DOUT_DATA'), n * 3, np.float32).reshape(n, 3)
            scores[:] = self.tools.forward(spectra.astype(np.float64), self.tools.load_quantized())
            if self.broken:
                scores[min(5, n - 1), 0] += 0.5
            self.runs.append(n)
        else:                              # the optimized one: the raw cube in, scores and class out
            h, w = self.words[self.regs['HEIGHT_DATA']], self.words[self.regs['WIDTH_DATA']]
            dins = {self.address(f'DIN{i}_DATA') for i in range(4)}
            assert len(dins) == 1, 'every port reads the cube'
            cube = self.view(dins.pop(), h * w * 120, np.uint16).reshape(h * w, 120)
            out = self.tools.classify_fixed(cube)
            if self.broken:
                out[min(5, len(out) - 1), 0] += 1
            self.view(self.address('DOUT_DATA'), h * w * 4, np.int32)[:] = out.reshape(-1)
            self.words[self.regs['CYCLES_DATA']] = 4 * h * w
            self.runs.append((h, w))
        self.done = True


@unittest.skipUnless(np is not None, 'needs numpy')
class SpectralPointwise1DNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('spectralpointwise1d_model', ROOT / 'tools/spectralpointwise1d_model.py')
        if not cls.model.quantized_path().exists():
            raise unittest.SkipTest('no quantized weights')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()        # a notebook's namespace (a cycle) may still hold cube.npy; Windows locks it
        self.tmp.cleanup()

    def prepare(self, kernel, regs, h, w, reference=True):
        """The board's folder: what the kernel's package (main.py, include.txt) and an image
        zip put there. The image is aegean pixels (real spectra, real classes) with the
        dropped bands at 65535, so a wrong band selection shows; with the PC's answer unless
        reference is False."""
        raw, labels = self.model.aegean_pixels()
        cube = np.rint(raw[:h * w]).astype(np.uint16).reshape(h, w, 120)
        kept, _, _ = self.model.read_preprocessing()
        cube[:, :, [b for b in range(120) if b not in kept]] = 65535
        board = self.dir / kernel
        image = board / 'aegean_unet'
        image.mkdir(parents=True)
        np.save(image / 'cube.npy', cube)
        np.save(image / 'labels.npy', labels[:h * w].reshape(h, w).astype(np.uint8))
        if reference:   # as python3 tools/spectralpointwise1d_model.py reference IMAGE writes it
            np.save(image / f'reference_{MODEL}.npy', self.model.classify_fixed(cube.reshape(-1, 120)).reshape(h, w, 4))
        (image / 'meta.json').write_text(json.dumps({'rgb_bands': [56, 67, 86]}))
        with zipfile.ZipFile(board / 'aegean_unet.zip', 'w') as z:
            for f in image.iterdir():
                z.write(f, f'aegean_unet/{f.name}')
        shutil.rmtree(image)
        name = kernel.upper()
        (board / f'x{kernel}_hw.h').write_text(''.join(
            f'#define X{name}_CONTROL_ADDR_{reg} 0x{offset:02x}\n' for reg, offset in regs.items()))
        include = ROOT / 'software/pynq' / kernel / 'include.txt'
        for line in include.read_text().splitlines():
            line = line.split('#')[0].strip()
            if line and not line.endswith('.ipynb'):
                shutil.copy2(ROOT / line, board)
        return board

    def run_notebook(self, notebook, board, kernel_name, regs, broken=False, max_bytes=None):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib."""
        kernel = FakeKernel(self.model, regs, broken, max_bytes)
        bitstreams = []
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: (bitstreams.append(path), types.SimpleNamespace(kernel=kernel))[1]
        pynq.allocate = kernel.allocate
        pynq.Clocks = types.SimpleNamespace(fclk0_mhz=187.5)
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
            for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), namespace)
        finally:
            os.chdir(cwd)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.assertEqual(bitstreams, [f'{kernel_name}.bit'])
        return namespace, kernel

    # ---- the golden kernel's notebook

    def golden(self, h, w, **kw):
        reference = kw.pop('reference', True)
        board = self.prepare(MODEL, GOLDEN_REGS, h, w, reference)
        return self.run_notebook(NOTEBOOK, board, MODEL, GOLDEN_REGS, **kw)

    def test_golden_notebook_classifies_the_image_like_the_numpy_model(self):
        ns, kernel = self.golden(50, 70)
        self.assertEqual(ns['NAME'], MODEL)
        self.assertIsNone(ns['strips'])
        self.assertEqual(kernel.runs, [256] + [50 * 70] * 6,
                         'the self-test, the whole image in one start, then 5 for the speed')
        self.assertEqual(ns['scores'].shape, (50, 70, 3))
        # the PC's integer model: this float kernel has its classes, to rounding
        self.assertGreater(ns['same_class'], 0.98)
        self.assertLess(float(np.abs(ns['scores']).max()), 1000, 'dropped bands reached the kernel')
        self.assertGreater((ns['classes'] == ns['labels']).mean(), 0.9)
        self.assertTrue((self.dir / MODEL / 'aegean_unet_fpga_scores.npy').is_file())

    def test_golden_notebook_uses_strips_when_the_image_does_not_fit(self):
        h, w = 300, 30                       # strips of 128, 128 and 44 rows
        ns, kernel = self.golden(h, w, max_bytes=h * w * 110 * 4 - 1)
        self.assertIsNotNone(ns['strips'])
        self.assertEqual(kernel.runs, [256, 128 * w, 128 * w, 44 * w] + [128 * w] * 5)
        self.assertGreater(ns['same_class'], 0.98)

    def test_golden_notebook_shows_a_wrong_kernel(self):
        ns, _ = self.golden(40, 40, broken=True)
        self.assertGreater(ns['error'], 1e-3 * (1 + np.abs(ns['want']).max()))

    # ---- the optimized kernels' notebook

    def optimized(self, h, w, kernel='spectralpointwise1d_opt', regs=OPT_REGS, **kw):
        reference = kw.pop('reference', True)
        board = self.prepare(kernel, regs, h, w, reference)
        return self.run_notebook(OPT_NOTEBOOK, board, kernel, regs, **kw)

    def test_optimized_notebook_classifies_the_image_bit_for_bit(self):
        ns, kernel = self.optimized(40, 60)
        self.assertEqual((ns['NAME'], ns['MODEL'], len(ns['DINS'])), ('spectralpointwise1d_opt', MODEL, 4))
        self.assertIsNone(ns['strips'])
        self.assertEqual(kernel.runs, [(1, 256)] + [(40, 60)] * 6,
                         'the self-test, the whole image in one start, then 5 for the speed')
        self.assertEqual(ns['differ'], 0)
        self.assertTrue(ns['close'].all())
        self.assertEqual(ns['close'].shape, (40, 60), 'every pixel, from the PC')
        self.assertGreater((ns['classes'] == ns['labels']).mean(), 0.9)

    def test_optimized_notebook_runs_the_zynq7020_kernel(self):
        """The PYNQ-Z2 build packs the same notebook: two read ports instead of four."""
        regs = {k: v for k, v in OPT_REGS.items() if k not in ('DIN2_DATA', 'DIN3_DATA')}
        kernel = 'spectralpointwise1d_opt_zynq7020'
        board = self.prepare(kernel, regs, 30, 30)
        fake_regs = dict(regs, DIN2_DATA=regs['DIN0_DATA'], DIN3_DATA=regs['DIN0_DATA'])
        ns, _ = self.run_notebook(OPT_NOTEBOOK, board, kernel, fake_regs)
        self.assertEqual((ns['NAME'], len(ns['DINS'])), (kernel, 2))
        self.assertEqual(ns['differ'], 0)
        self.assertTrue(ns['close'].all())

    def test_optimized_notebook_uses_strips_when_the_cube_does_not_fit(self):
        h, w = 300, 20                       # strips of 128, 128 and 44 rows
        ns, kernel = self.optimized(h, w, max_bytes=h * w * 120 * 2 - 1)
        self.assertIsNotNone(ns['strips'])
        self.assertEqual(kernel.runs, [(1, 256), (128, w), (128, w), (44, w)] + [(128, w)] * 5)
        self.assertTrue(ns['close'].all())

    def test_optimized_notebook_shows_a_wrong_kernel(self):
        ns, _ = self.optimized(30, 30, broken=True)
        self.assertGreater(ns['differ'], 0)
        self.assertFalse(ns['close'].all())

    def test_optimized_notebook_without_a_reference_checks_the_labels_only(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ns, _ = self.optimized(30, 30, reference=False)
        self.assertIsNone(ns['close'])
        self.assertIn(f'no reference_{MODEL}.npy', printed.getvalue())
        self.assertIn('accuracy against the labels', printed.getvalue())


if __name__ == '__main__':
    unittest.main()
