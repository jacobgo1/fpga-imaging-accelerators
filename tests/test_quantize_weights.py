"""tools/quantize_weights.py: BatchNorm folding, the quantization itself, the headers."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = sorted((ROOT / 'weights').rglob('*.pt'))
CXX = shutil.which('g++') or shutil.which('clang++')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(fragment):
    return next(p for p in CHECKPOINTS if fragment in p.name)


@unittest.skipIf(np is None, 'the quantizer needs numpy')
@unittest.skipUnless(CHECKPOINTS, 'no checkpoints in weights/')
class QuantizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.tool = load_module('quantize_weights', ROOT / 'tools/quantize_weights.py')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))
        cls.tmp.cleanup()

    def run_tool(self, *args):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/quantize_weights.py'), *map(str, args),
                                 '--out', str(self.out)], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_folding_is_the_batchnorm_formula(self):
        # conv then BN = conv with w' = w * s and b' = (b - mean) * s + beta, s = gamma / sqrt(var + eps)
        _, _, tensors = self.tool.ew.load(checkpoint('justoliunet_bn'))
        folded, pairs = self.tool.fold_batchnorm(tensors)
        self.assertEqual(pairs, {'conv1': 'bn1'})
        s = tensors['bn1.weight'] / np.sqrt(tensors['bn1.running_var'] + 1e-5)
        w = tensors['conv1.weight']
        np.testing.assert_allclose(folded['conv1.weight'], w * s.reshape((-1,) + (1,) * (w.ndim - 1)),
                                   rtol=1e-6, atol=1e-7)
        b = (tensors.get('conv1.bias', 0.0) - tensors['bn1.running_mean']) * s + tensors['bn1.bias']
        np.testing.assert_allclose(folded['conv1.bias'], b, rtol=1e-6, atol=1e-7)

    def test_every_batchnorm_is_folded_in_every_model(self):
        for path in CHECKPOINTS:
            _, _, tensors = self.tool.ew.load(path)
            folded, pairs = self.tool.fold_batchnorm(tensors)
            with self.subTest(model=path.parent.name):
                bns = {name.rpartition('.')[0] for name in tensors if name.endswith('running_var')}
                self.assertEqual(set(pairs.values()), bns)
                self.assertFalse([n for n in folded if 'running_' in n])
                for conv in pairs:  # conv layers without a bias get one from the BN
                    self.assertIn(f'{conv}.bias', folded)

    def test_power_of_two_quantization_is_exact_where_it_does_not_clip(self):
        x = np.random.default_rng(0).normal(0, 0.3, (16, 9))
        q, frac, deq = self.tool.quantize(x, 8, 'pow2', False)
        self.assertEqual(q.dtype, np.int8)
        step = 2.0 ** -frac
        np.testing.assert_array_equal(deq, q * step)
        inside = np.abs(x) < 127 * step
        self.assertTrue(np.all(np.abs(x - deq)[inside] <= step / 2 + 1e-12))
        no_clip = int(np.floor(np.log2(127 / np.max(np.abs(x)))))
        mse = lambda f: np.sum((np.clip(np.round(x * 2.0 ** f), -128, 127) * 2.0 ** -f - x) ** 2)
        self.assertLessEqual(mse(frac), mse(no_clip))

    def test_narrow_and_per_channel_and_float_scales(self):
        x = np.random.default_rng(1).normal(0, 1, (4, 3, 5)) * np.array([1, 10, 0.1, 3])[:, None, None]
        q4, _, _ = self.tool.quantize(x, 4, 'pow2', False)
        self.assertGreaterEqual(q4.min(), -8)
        self.assertLessEqual(q4.max(), 7)
        _, fracs, deq_pc = self.tool.quantize(x, 8, 'pow2', True)
        self.assertEqual(fracs.shape, (4,))
        _, _, deq_pt = self.tool.quantize(x, 8, 'pow2', False)
        self.assertGreater(self.tool.sqnr_db(x[2], deq_pc[2]), self.tool.sqnr_db(x[2], deq_pt[2]),
                           'the small channel gains from its own scale')
        q, scales, _ = self.tool.quantize(x, 8, 'float', True)
        for row in q.reshape(4, -1):  # each channel reaches one end of the range exactly
            self.assertTrue(row.max() == 127 or row.min() == -128, row)
        self.assertEqual(scales.dtype, np.float32)

    def test_the_binding_side_reaches_its_end_of_the_range(self):
        q, _, _ = self.tool.quantize(np.array([-0.8, -0.1, 0.0, 0.3, 0.5]), 8, 'float', False)
        self.assertEqual((q.min(), q.max()), (-128, 80))       # negative side binds: -0.8 -> -128
        q, _, _ = self.tool.quantize(np.array([-0.2, 0.0, 0.9]), 8, 'float', False)
        self.assertEqual((q.min(), q.max()), (-28, 127))       # positive side binds: 0.9 -> 127
        q, _, _ = self.tool.quantize(np.array([0.375, 1.16]), 8, 'float', False)
        self.assertEqual(q.max(), 127)                          # one sign only: still uses its end
        q, frac, _ = self.tool.quantize(np.array([-1.0, 0.25]), 8, 'pow2', False)
        self.assertEqual((q.min(), int(frac)), (-128, 7))       # -1.0 * 2^7 = -128 fits; 127 would not

    def test_more_bits_mean_less_noise(self):
        _, _, tensors = self.tool.ew.load(checkpoint('sp_unet_small'))
        x = self.tool.fold_batchnorm(tensors)[0]['double2.block.0.weight']
        sqnr = {bits: self.tool.sqnr_db(x, self.tool.quantize(x, bits, 'pow2', False)[2]) for bits in (4, 8, 16)}
        self.assertLess(sqnr[4], sqnr[8])
        self.assertLess(sqnr[8], sqnr[16])

    @unittest.skipIf(CXX is None, 'needs g++ or clang++')
    def test_headers_compile_and_hold_the_npz_values(self):
        self.run_tool()                                                  # int8, all models
        self.run_tool(checkpoint('justoliunet_bn'), '--per-channel')
        self.run_tool(checkpoint('waveletcnn'), '--scale', 'float', '--bits', '16')
        headers = sorted(self.out.glob('*/*.hpp'))
        self.assertEqual(len(headers), len(CHECKPOINTS) + 2)
        overview = (self.out / 'README.md').read_text()
        for header in headers:
            self.assertIn(f'[{header.stem}]', overview)
        stem = 'justoliunet_bn_int8_perchannel'
        saved = np.load(self.out / 'justoliunet_bn' / f'{stem}.npz')
        source = (''.join(f'#include "{h.parent.name}/{h.name}"\n' for h in headers) + '#include <cstdio>\n'
                  'int main() { for (int o = 0; o < 6; o++) std::printf("%d %d ", '
                  f'(int){stem}::conv1_weight[o][0][5], {stem}::conv1_weight_frac[o]); return 0; }}\n')
        (self.out / 'check.cpp').write_text(source)
        exe = self.out / 'check.exe'
        build = subprocess.run([CXX, '-std=c++14', '-I', str(self.out), str(self.out / 'check.cpp'), '-o', str(exe)],
                               capture_output=True, text=True)
        self.assertEqual(build.returncode, 0, build.stderr[-2000:])
        printed = [int(v) for v in subprocess.run([str(exe)], capture_output=True, text=True).stdout.split()]
        self.assertEqual(printed[0::2], saved['conv1.weight'][:, 0, 5].tolist())
        self.assertEqual(printed[1::2], saved['conv1.weight.frac'].tolist())


if __name__ == '__main__':
    unittest.main()
