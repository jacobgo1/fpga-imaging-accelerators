"""justoliunet_opt and justoliunet_bn_opt (ZCU104) and their Zynq-7020 versions: the integer
model against the trained model, every optimized C++ kernel against the integer model (the
same answers on both chips) on the labelled aegean pixels, the reference files for the
ZedBoard's classify, and the ZCU104 notebook end to end against a fake pynq whose kernel is
the integer model.

classify_fixed (tools/justoliunet_model.py) is what the kernels must write, bit for bit;
here it is checked against the trained fp32 model on real pixels, and the kernels,
compiled with plain g++, against it on the same pixels. The kernels' testbenches check
them against a plain-loop integer reference of their own and the float golden network.
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
KERNELS = {'justoliunet': 'justoliunet_opt', 'justoliunet_bn': 'justoliunet_bn_opt'}
# Every optimized kernel: (model, its call on din, dout, height, width, cycles). The ZCU104's
# read the cube through two ports, the Zynq-7000s' through one.
OPTIMIZED = {'justoliunet_opt': ('justoliunet', 'justoliunet_opt(d, d, dout.data(), 1, n, cycles)'),
             'justoliunet_bn_opt': ('justoliunet_bn', 'justoliunet_bn_opt(d, d, dout.data(), 1, n, cycles)'),
             'justoliunet_opt_zynq7020': ('justoliunet', 'justoliunet_opt_zynq7020(d, dout.data(), 1, n, cycles)'),
             'justoliunet_bn_opt_zynq7020': ('justoliunet_bn', 'justoliunet_bn_opt_zynq7020(d, dout.data(), 1, n, cycles)'),
             'justoliunet_opt_zynq7030': ('justoliunet', 'justoliunet_opt_zynq7030(d, dout.data(), 1, n, cycles)'),
             'justoliunet_bn_opt_zynq7030': ('justoliunet_bn', 'justoliunet_bn_opt_zynq7030(d, dout.data(), 1, n, cycles)')}
NOTEBOOK = ROOT / 'software/pynq/justoliunet_opt/justoliunet_opt.ipynb'
# Register offsets for the fake kernel, in the layout Vitis HLS gives the kernel.
REGS = {'AP_CTRL': 0x00, 'DIN0_DATA': 0x10, 'DIN1_DATA': 0x1c, 'DOUT_DATA': 0x28,
        'HEIGHT_DATA': 0x34, 'WIDTH_DATA': 0x3c, 'CYCLES_DATA': 0x44}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(np is not None, 'needs numpy')
class JustoLiuNetOptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('justoliunet_model', ROOT / 'tools/justoliunet_model.py')
        cls.raw, cls.labels = cls.model.aegean_pixels()
        kept, mean, inv_std = cls.model.read_preprocessing()
        cls.x = cls.model.preprocess(cls.raw, kept, mean, inv_std)
        missing = [m for m in KERNELS if not cls.model.quantized_path(m).exists()]
        if missing:
            raise unittest.SkipTest(f'no quantized weights for {missing}: '
                                    'python tools/quantize_weights.py weights/fp32/justoliunet weights/fp32/justoliunet_bn')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def test_formats_header_is_up_to_date(self):
        """src/hls/justoliunet_opt/justoliunet_opt_formats.hpp comes from the model's formats."""
        self.assertEqual(self.model.FORMATS_HEADER.read_text(), self.model.formats_header(),
                         'run: python tools/justoliunet_model.py formats')

    def test_integer_model_keeps_the_trained_classes(self):
        """12-bit input, uint8 activations: as good as int8 weights with float activations."""
        for model, least in (('justoliunet', 0.995), ('justoliunet_bn', 0.985)):
            with self.subTest(model=model):
                fp32 = self.model.forward_fp32(self.x, model).argmax(1)
                fixed = self.model.classify_fixed(self.raw, model)
                self.assertGreater((fixed[:, 3] == fp32).mean(), least)
                self.assertLess(abs((fixed[:, 3] == self.labels).mean() - (fp32 == self.labels).mean()), 0.01)
                # the class lane is the first highest score
                np.testing.assert_array_equal(fixed[:, 3], fixed[:, :3].argmax(1))

    def test_integer_z_score_is_within_one_unit_of_the_float_one(self):
        kept, mean, inv_std = self.model.read_preprocessing()
        exact = (self.raw[:, kept].astype(np.float64) - mean) * inv_std * 2.0 ** self.model.IN_FRAC
        exact = np.clip(exact, -self.model.IN_MAX, self.model.IN_MAX)
        self.assertLessEqual(np.abs(self.model.preprocess_fixed(self.raw) - exact).max(), 1.0)

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_cpp_kernels_are_the_integer_model_bit_for_bit(self):
        n = len(self.raw)                                  # 10 275 pixels: one row of them
        cube = self.raw.astype(np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cube.tofile(tmp / 'in.bin')
            for kernel, (model, call) in OPTIMIZED.items():
                with self.subTest(kernel=kernel):
                    source_dir = ROOT / 'src/hls' / kernel
                    source = tmp / f'run_{kernel}.cpp'
                    source.write_text(f"""
#include "{kernel}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int n = atoi(argv[3]);
    std::vector<cube_word_t> din(n * WORDS_PER_PIXEL);
    std::vector<score_word_t> dout(n);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(cube_word_t), din.size(), f) != din.size()) return 1;
    fclose(f);
    uint32_t cycles = 0;
    cube_word_t *d = din.data();     // hls::burst_maxi takes a plain pointer
    {call};
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(score_word_t), dout.size(), f); fclose(f);
}}
""")
                    exe = tmp / f'run_{kernel}.exe'
                    subprocess.run([CXX, '-std=c++14', '-O2', f'-I{source_dir}', f'-I{ROOT / "src/common"}',
                                    str(source), str(source_dir / f'{kernel}.cpp'), '-o', str(exe)], check=True)
                    out = tmp / f'out_{kernel}.bin'
                    subprocess.run([str(exe), str(tmp / 'in.bin'), str(out), str(n)], check=True)
                    cpp = np.fromfile(out, dtype=np.int32).reshape(n, 4)
                    np.testing.assert_array_equal(cpp, self.model.classify_fixed(self.raw, model))

    def test_reference_files_are_the_integer_model(self):
        """tools/justoliunet_model.py reference IMAGE: what the kernels must write for
        IMAGE/cube.npy, per model, in chunks that need not divide the image."""
        h, w = 23, 41
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / 'cube.npy', self.raw[:h * w].astype(np.uint16).reshape(h, w, 120))
            written = self.model.reference(tmp, chunk=100)
            self.assertEqual(sorted(p.name for p in written),
                             ['reference_justoliunet.npy', 'reference_justoliunet_bn.npy'])
            for model in KERNELS:
                got = np.load(Path(tmp) / f'reference_{model}.npy')
                self.assertEqual((got.shape, got.dtype), ((h, w, 4), np.int32))
                np.testing.assert_array_equal(got.reshape(-1, 4), self.model.classify_fixed(self.raw[:h * w], model))

    def test_zynq7020_kernels_have_the_opu_program(self):
        """A zedboard build packs software/opu/KERNEL/ (main.py): the generic classify.c."""
        for kernel in ('justoliunet_opt_zynq7020', 'justoliunet_bn_opt_zynq7020'):
            with self.subTest(kernel=kernel):
                include = ROOT / 'software/opu' / kernel / 'include.txt'
                files = [line.split('#')[0].strip() for line in include.read_text().splitlines()]
                files = [f for f in files if f]
                self.assertEqual(files, ['software/opu/justounetsimple_opt/classify.c'])
                self.assertTrue((ROOT / files[0]).is_file())


class FakeBuffer(np.ndarray if np is not None else object):
    """pynq.allocate's buffer: an array with a physical address."""

    def flush(self): pass
    def invalidate(self): pass


class FakeOptKernel:
    """The HLS block's AXI-Lite registers; on start it classifies the height x width cube at
    din0 with the integer model and writes the scores and classes at dout (anywhere inside a
    buffer: strips write part-way into the scores). Allocations larger than max_bytes fail,
    as when PYNQ's CMA is short."""

    def __init__(self, model, tools, broken=False, max_bytes=None):
        self.model, self.tools, self.broken, self.max_bytes = model, tools, broken, max_bytes
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
        offset = REGS[name]
        return self.words.get(offset, 0) | (self.words.get(offset + 4, 0) << 32)

    def view(self, address, count, dtype):
        for start, buffer in self.buffers.items():
            if start <= address and address + count * np.dtype(dtype).itemsize <= start + buffer.nbytes:
                assert buffer.dtype == dtype
                return np.asarray(buffer).reshape(-1)[(address - start) // np.dtype(dtype).itemsize:][:count]
        raise AssertionError(f'0x{address:x}: no buffer holds {count} {np.dtype(dtype)} there')

    def read(self, offset):
        if offset == REGS['AP_CTRL']:
            done, self.done = self.done, False
            return (0x2 if done else 0) | 0x4
        return self.words.get(offset, 0)

    def write(self, offset, value):
        if offset == REGS['AP_CTRL'] and value & 1:
            h, w = self.words[REGS['HEIGHT_DATA']], self.words[REGS['WIDTH_DATA']]
            assert self.address('DIN0_DATA') == self.address('DIN1_DATA'), 'both ports read the cube'
            cube = self.view(self.address('DIN0_DATA'), h * w * 120, np.uint16).reshape(h * w, 120)
            out = self.tools.classify_fixed(cube, self.model)
            if self.broken:
                out[min(5, len(out) - 1), 0] += 1
            self.view(self.address('DOUT_DATA'), h * w * 4, np.int32)[:] = out.reshape(-1)
            self.words[REGS['CYCLES_DATA']] = 18 * h * w
            self.runs.append((h, w))
            self.done = True
        else:
            self.words[offset] = value


@unittest.skipUnless(np is not None, 'needs numpy')
class JustoLiuNetOptNotebookTests(unittest.TestCase):
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

    def prepare(self, model, h, w, reference=True):
        """The board's folder: what the kernel's package (main.py, include.txt) and an image
        zip put there. The image is aegean pixels (real spectra, real classes) with the
        dropped bands at 65535, so a wrong band selection shows; with the PC's answer for
        the model unless reference is False."""
        if not self.model.quantized_path(model).exists():
            self.skipTest(f'no quantized weights for {model}')
        raw, labels = self.model.aegean_pixels()
        cube = np.rint(raw[:h * w]).astype(np.uint16).reshape(h, w, 120)
        kept, _, _ = self.model.read_preprocessing()
        cube[:, :, [b for b in range(120) if b not in kept]] = 65535
        board = self.dir / model
        image = board / 'aegean_unet'
        image.mkdir(parents=True)
        np.save(image / 'cube.npy', cube)
        np.save(image / 'labels.npy', labels[:h * w].reshape(h, w).astype(np.uint8))
        if reference:   # as python3 tools/justoliunet_model.py reference IMAGE writes it
            np.save(image / f'reference_{model}.npy',
                    self.model.classify_fixed(cube.reshape(-1, 120), model).reshape(h, w, 4))
        (image / 'meta.json').write_text(json.dumps({'rgb_bands': [56, 67, 86]}))
        with zipfile.ZipFile(board / 'aegean_unet.zip', 'w') as z:
            for f in image.iterdir():
                z.write(f, f'aegean_unet/{f.name}')
        shutil.rmtree(image)
        name = KERNELS[model].upper()
        (board / f'x{KERNELS[model]}_hw.h').write_text(''.join(
            f'#define X{name}_CONTROL_ADDR_{reg} 0x{offset:02x}\n' for reg, offset in REGS.items()))
        include = ROOT / 'software/pynq' / KERNELS[model] / 'include.txt'
        for line in include.read_text().splitlines():
            line = line.split('#')[0].strip()
            if line and not line.endswith('.ipynb'):
                shutil.copy2(ROOT / line, board)
        return board

    def run_notebook(self, board, model, broken=False, max_bytes=None):
        """Every code cell of the notebook, in order, with fake pynq and matplotlib."""
        kernel = FakeOptKernel(model, self.model, broken, max_bytes)
        bitstreams = []
        pynq = types.ModuleType('pynq')
        pynq.Overlay = lambda path: (bitstreams.append(path), types.SimpleNamespace(kernel=kernel))[1]
        pynq.allocate = kernel.allocate
        pynq.Clocks = types.SimpleNamespace(fclk0_mhz=200.0)
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
        self.assertEqual(bitstreams, [f'{KERNELS[model]}.bit'])
        return namespace, kernel

    def test_notebook_classifies_the_image_bit_for_bit(self):
        for model in KERNELS:
            with self.subTest(model=model):
                ns, kernel = self.run_notebook(self.prepare(model, 40, 60), model)
                self.assertEqual((ns['NAME'], ns['MODEL']), (KERNELS[model], model))
                self.assertIsNone(ns['strips'])
                self.assertEqual(kernel.runs, [(1, 256)] + [(40, 60)] * 6,
                                 'the self-test, the whole image in one start, then 5 for the speed')
                self.assertEqual(ns['differ'], 0)
                self.assertTrue(ns['close'].all())
                self.assertEqual(ns['close'].shape, (40, 60), 'every pixel, from the PC')
                self.assertGreater((ns['classes'] == ns['labels']).mean(), 0.9)
                self.assertTrue((self.dir / model / 'aegean_unet_fpga_scores.npy').is_file())

    def test_notebook_uses_strips_when_the_cube_does_not_fit(self):
        h, w = 300, 20                       # strips of 128, 128 and 44 rows
        ns, kernel = self.run_notebook(self.prepare('justoliunet', h, w), 'justoliunet',
                                       max_bytes=h * w * 120 * 2 - 1)
        self.assertIsNotNone(ns['strips'])
        self.assertEqual(kernel.runs, [(1, 256), (128, w), (128, w), (44, w)] + [(128, w)] * 5)
        self.assertTrue(ns['close'].all())

    def test_notebook_shows_a_wrong_kernel(self):
        ns, _ = self.run_notebook(self.prepare('justoliunet', 30, 30), 'justoliunet', broken=True)
        self.assertGreater(ns['differ'], 0)
        self.assertFalse(ns['close'].all())

    def test_notebook_without_a_reference_checks_the_labels_only(self):
        # The model is not run on the ARM over the image: without the PC's answer there is
        # no comparison with it, only the labels.
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            ns, _ = self.run_notebook(self.prepare('justoliunet', 30, 30, reference=False), 'justoliunet')
        self.assertIsNone(ns['close'])
        self.assertIn('no reference_justoliunet.npy', printed.getvalue())
        self.assertIn('accuracy against the labels', printed.getvalue())
        self.assertGreater((ns['classes'] == ns['labels']).mean(), 0.9)


if __name__ == '__main__':
    unittest.main()
