"""justounetsimple: the numpy models against the C++ kernels. The board notebook, shared by every
model, is tests/test_notebooks.py's; the OPU program, test_opu_classify.py's.
"""
import gc
from pathlib import Path
import re
import sys
import tempfile
import unittest

from helpers import CXX, ROOT, kernel_dir, load_module, raw_cube, run_kernel

try:
    import numpy as np
except ImportError:
    np = None

OPT_SOURCE = kernel_dir('justounetsimple_opt')


@unittest.skipIf(np is None, 'needs numpy')
class JustoUNetSimpleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('justounetsimple_model', ROOT / 'tools/justounetsimple_model.py')
        cls.patching = load_module('patching', ROOT / 'tools/patching.py')
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

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_numpy_model_matches_the_cpp_kernel(self):
        x = np.random.default_rng(3).uniform(-3, 3, (32, 32, 110)).astype(np.float32)
        x.tofile(self.dir / 'in.bin')
        cpp = run_kernel(self.dir, 'justounetsimple', """
#include "justounetsimple.hpp"
#include <cstdio>
static float din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS];
static float dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES];
int main(int, char** argv) {
    FILE* f = fopen(argv[1], "rb"); if (fread(din, sizeof din, 1, f) != 1) return 1; fclose(f);
    justounetsimple(din, dout);
    f = fopen(argv[2], "wb"); fwrite(dout, sizeof dout, 1, f); fclose(f);
}
""", self.dir / 'in.bin', np.float32, (32, 32, 3))
        want = self.model.forward_patch(x, self.layers)
        np.testing.assert_allclose(cpp, want, rtol=1e-3, atol=1e-3)
        self.assertGreater(float(want.max() - want.min()), 1.0, 'the reference should not be constant')

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_fixed_point_model_matches_the_optimized_cpp_kernel(self):
        """classify_fixed predicts the board's answers: it must be the optimized kernels'
        arithmetic bit for bit (the ZCU104's and the Zynq-7000s'), the z-score and the edge
        patches included."""
        h, w = 37, 45                                     # 2 x 2 patches, the edge ones partial
        cube = raw_cube(self.model, h, w, seed=3, counts=True).astype(np.uint16)
        cube.tofile(self.dir / 'in.bin')
        want = self.model.classify_fixed(cube)
        kernels = {'justounetsimple_opt': 'uint32_t cycles; justounetsimple_opt(d, d, d, d, dout.data(), h, w, cycles)',
                   'justounetsimple_opt_tiled': 'uint32_t cycles; justounetsimple_opt_tiled(d, d, d, d, dout.data(), h, w, cycles)',
                   'justounetsimple_opt_zynq7020': 'justounetsimple_opt_zynq7020(d, d, dout.data(), h, w)',
                   'justounetsimple_opt_zynq7030': 'justounetsimple_opt_zynq7030(d, d, d, d, dout.data(), h, w)'}
        for name, call in kernels.items():
            with self.subTest(kernel=name):
                cpp = run_kernel(self.dir, name, f"""
#include "{name}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    std::vector<cube_word_t> din(h * w * CUBE_BANDS / BANDS_PER_WORD);
    std::vector<score_word_t> dout(h * w);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(cube_word_t), din.size(), f) != din.size()) return 1;
    fclose(f);
    cube_word_t *d = din.data();   // hls::burst_maxi takes a plain pointer
    {call};
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(score_word_t), dout.size(), f); fclose(f);
}}
""", self.dir / 'in.bin', np.int32, (h, w, 4))
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
        cube = raw_cube(self.model, 16, 16, counts=True)
        kept, mean, inv_std = self.model.read_preprocessing()
        floats = self.model.quantize_input(self.model.preprocess(cube, kept, mean, inv_std))
        ints = self.model.preprocess_fixed(cube.astype(np.uint16))
        self.assertLessEqual(int(np.abs(ints.astype(int) - floats).max()), 1)

    def test_prep_header_is_up_to_date(self):
        """zscore_int8.hpp is generated from mu_sd.txt: regenerate it when that changes."""
        prep = load_module('preprocessing', ROOT / 'tools/preprocessing.py')
        self.assertEqual(prep.HEADER.read_text(encoding='utf-8'), prep.header(),
                         'run: python tools/preprocessing.py')

    def test_model_uses_the_kernel_formats(self):
        """The numpy model and tools/models.py hard-code what justounetsimple_opt.hpp defines."""
        header = (OPT_SOURCE / 'justounetsimple_opt.hpp').read_text(encoding='utf-8')

        def define(name):
            return int(re.search(rf'constexpr int {name} = (\d+);', header).group(1))

        models = load_module('models', ROOT / 'tools/models.py')
        self.assertEqual(models.MODELS['justounetsimple']['out_frac'], define('SCORE_FRAC_BITS'))
        self.assertEqual(define('CUBE_BANDS'), 120)
        self.assertIn('hls::vector<int32_t, 4> score_word_t', header)
        for model_name, kernel_name in (('IN_FRAC', 'ZSCORE_FRAC_BITS'), ('ACT_FRAC', 'ACTIVATION_FRAC_BITS'),
                                        ('OUT_FRAC', 'SCORE_FRAC_BITS')):
            self.assertEqual(getattr(self.model, model_name), define(kernel_name), kernel_name)