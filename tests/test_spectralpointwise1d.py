"""spectralpointwise1d (golden) and spectralpointwise1d_opt with its Zynq-7030 and Zynq-7020
versions: the numpy models against the trained model, every C++ kernel against them on the
labelled aegean pixels (the optimized ones bit for bit, the same answers on every board) and
the OPU program. The board notebooks, shared by every model, are tests/test_notebooks.py's.

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
# Register offsets for the fake kernels, in the layout Vitis HLS gives them.


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

    def test_petalinux_kernels_have_the_opu_program(self):
        """A zedboard or zynq7030 build packs software/opu/KERNEL/ (main.py): the generic classify.c."""
        for kernel in ('spectralpointwise1d_opt_zynq7020', 'spectralpointwise1d_opt_zynq7030'):
            with self.subTest(kernel=kernel):
                include = ROOT / 'software/opu' / kernel / 'include.txt'
                files = [line.split('#')[0].strip() for line in include.read_text().splitlines()]
                files = [f for f in files if f]
                self.assertEqual(files, ['software/opu/justounetsimple_opt/classify.c'])
                self.assertTrue((ROOT / files[0]).is_file())

if __name__ == '__main__':
    unittest.main()
