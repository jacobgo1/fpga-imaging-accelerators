"""justoliunet_opt and justoliunet_bn_opt (ZCU104) and their Zynq-7020 versions: the integer
model against the trained model, every optimized C++ kernel against the integer model (the
same answers on both chips) on the labelled aegean pixels. The board notebook, shared by every
model, is tests/test_notebooks.py's.

classify_fixed (tools/justoliunet_model.py) is what the kernels must write, bit for bit;
here it is checked against the trained fp32 model on real pixels, and the kernels,
compiled with plain g++, against it on the same pixels. The kernels' testbenches check
them against a plain-loop integer reference of their own and the float golden network.
"""
from pathlib import Path
import sys
import tempfile
import unittest

from helpers import CXX, ROOT, load_module, run_kernel, skip_without_aegean

try:
    import numpy as np
except ImportError:
    np = None

KERNELS = {'justoliunet': 'justoliunet_opt', 'justoliunet_bn': 'justoliunet_bn_opt'}
# Every optimized kernel: (model, its call on din, dout, height, width, cycles). The ZCU104's
# read the cube through two ports, the Zynq-7000s' through one.
OPTIMIZED = {'justoliunet_opt': ('justoliunet', 'justoliunet_opt(d, d, dout.data(), 1, n, cycles)'),
             'justoliunet_bn_opt': ('justoliunet_bn', 'justoliunet_bn_opt(d, d, dout.data(), 1, n, cycles)'),
             'justoliunet_opt_zynq7020': ('justoliunet', 'justoliunet_opt_zynq7020(d, dout.data(), 1, n, cycles)'),
             'justoliunet_bn_opt_zynq7020': ('justoliunet_bn', 'justoliunet_bn_opt_zynq7020(d, dout.data(), 1, n, cycles)'),
             'justoliunet_opt_zynq7030': ('justoliunet', 'justoliunet_opt_zynq7030(d, dout.data(), 1, n, cycles)'),
             'justoliunet_bn_opt_zynq7030': ('justoliunet_bn', 'justoliunet_bn_opt_zynq7030(d, dout.data(), 1, n, cycles)')}
# Register offsets for the fake kernel, in the layout Vitis HLS gives the kernel.


@unittest.skipUnless(np is not None, 'needs numpy')
class JustoLiuNetOptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_without_aegean()
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
        """justoliunet_opt_formats.hpp comes from the model's formats."""
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
                    cpp = run_kernel(tmp, kernel, f"""
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
""", tmp / 'in.bin', np.int32, (n, 4))
                    np.testing.assert_array_equal(cpp, self.model.classify_fixed(self.raw, model))


if __name__ == '__main__':
    unittest.main()
