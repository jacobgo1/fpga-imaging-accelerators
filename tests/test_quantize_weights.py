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

    def test_folding_matches_the_justoliunet_exporter(self):
        export = load_module('export_justoliunet', ROOT / 'tools/export_justoliunet.py')
        path = checkpoint('justoliunet_bn')
        convs, _ = export.load_layers(path)
        _, _, tensors = self.tool.ew.load(path)
        folded, pairs = self.tool.fold_batchnorm(tensors)
        self.assertEqual(pairs, {'conv1': 'bn1'})
        np.testing.assert_allclose(folded['conv1.weight'], convs[0][0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(folded['conv1.bias'], convs[0][1], rtol=1e-6, atol=1e-7)

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
        self.assertEqual(np.abs(q.reshape(4, -1)).max(axis=1).tolist(), [127] * 4)
        self.assertEqual(scales.dtype, np.float32)

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
