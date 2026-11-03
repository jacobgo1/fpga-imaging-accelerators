"""justoliunet and justoliunet_bn: the numpy model against the trained model and the C++ kernels.

The numpy model (tools/justoliunet_model.py) is checked against the old PC reference of
the aegean pixels and against the labels; the C++ kernels, compiled with plain g++,
against the numpy model on the same real pixels. The kernels' testbenches compare them
with a float network built from the same golden layers (src/golden/layers.hpp);
this test is the independent check of those layers. The board notebook, shared by every
model, is tests/test_notebooks.py's.
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

MODELS = ('justoliunet', 'justoliunet_bn')
AP_CTRL, DIN, DOUT, N_REG = 0x00, 0x10, 0x1c, 0x28


def quantized(model):
    return ROOT / 'weights/quantized' / model / f'{model}_int8.npz'


@unittest.skipUnless(np is not None, 'needs numpy')
class JustoLiuNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        skip_without_aegean()
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
                    cpp = run_kernel(tmp, model, f"""
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
""", tmp / 'in.bin', np.float32, (len(pixels), 3))
                    want = self.model.forward(pixels.astype(np.float64), self.model.load_quantized(model))
                    np.testing.assert_allclose(cpp, want, rtol=1e-4, atol=1e-4)
                    self.assertTrue((cpp.argmax(1) == want.argmax(1)).all())


if __name__ == '__main__':
    unittest.main()
