"""justoliunet: the image tool (prepare/compare), and generated files vs exporter."""
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy
except ImportError:
    numpy = None

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'weights/fp32/justoliunet/justoliunet_pixel1d_e31_miou0-908_K110.pt'
BANDS = 120


FAKE_HYPSO = '''
import numpy as np


class _Cube:
    def __init__(self, array):
        self._array = array

    def to_numpy(self):
        return self._array


class Hypso2:
    """Stands in for hypso.Hypso2: the L1a cube comes from <capture>.cube.npy."""
    def __init__(self, path):
        self.l1a_cube = _Cube(np.load(str(path) + '.cube.npy'))
'''


@unittest.skipIf(numpy is None, 'image tool needs numpy')
@unittest.skipUnless(CHECKPOINT.exists(), 'checkpoint not present')
class ImageToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        sys.path.insert(0, str(ROOT / 'tools'))
        self.rng = numpy.random.default_rng(0)
        raw = self.rng.uniform(100, 4000, (12, 9, BANDS)).astype(numpy.float32)
        numpy.save(self.dir / 'cube.npy', raw.transpose(2, 0, 1))
        numpy.save(self.dir / 'labels.npy', self.rng.integers(-1, 3, (12, 9)))

    def tearDown(self):
        sys.path.remove(str(ROOT / 'tools'))
        self.tmp.cleanup()

    def tool(self, *args, env=None):
        return subprocess.run([sys.executable, str(ROOT / 'tools/justoliunet_image.py'), *map(str, args)],
                              capture_output=True, text=True, cwd=ROOT, env=env)

    def prepare(self, cube='cube.npy', labels='labels.npy', env=None):
        result = self.tool('prepare', self.dir / cube, '--labels', self.dir / labels,
                           '--crop', 2, 1, 8, 6, '--step', 2, '--out', self.dir / 'img', env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.dir / 'img', numpy.load(self.dir / 'img/reference_logits.npy'), result.stdout

    def test_matching_fpga_output_passes_and_draws_maps(self):
        img, reference, _ = self.prepare()
        self.assertEqual(len(reference), 4 * 3)
        first = (img / 'pixels.txt').read_text().splitlines()[1].split()
        self.assertEqual(len(first), BANDS, 'the board gets raw spectra')
        numpy.savetxt(img / 'fpga_logits.txt', reference.astype(numpy.float32), fmt='%.9g')
        result = self.tool('compare', img)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('MATCH', result.stdout)
        self.assertIn('vs labels', result.stdout)
        for name in ('rgb', 'capture_rgb', 'capture_labels', 'overview', 'fpga_classes',
                     'reference_classes', 'mismatch', 'labels'):
            self.assertEqual((img / f'{name}.png').read_bytes()[:4], b'\x89PNG')

    def test_wrong_or_missing_pixels_fail(self):
        img, reference, _ = self.prepare()
        wrong = reference.copy()
        wrong[3, 0] += 0.01 + 0.01 * abs(wrong[3, 0])  # well outside the relative tolerance
        numpy.savetxt(img / 'fpga_logits.txt', wrong, fmt='%.9g')
        self.assertEqual(self.tool('compare', img).returncode, 1)
        numpy.savetxt(img / 'fpga_logits.txt', reference[:5], fmt='%.9g')
        result = self.tool('compare', img)
        self.assertEqual(result.returncode, 1)
        self.assertIn('5 of 12 pixels', result.stdout)

    def test_prepared_training_image_gets_the_network_output_it_had_in_training(self):
        from export_justoliunet import forward, load_layers, synthetic_spectra
        prepared = synthetic_spectra(self.rng, 12 * 9, 110).reshape(12, 9, 110).astype(numpy.float32)
        numpy.save(self.dir / 'data3.npy', prepared)
        _, reference, stdout = self.prepare('data3.npy')
        self.assertIn('prepared', stdout)
        convs, fc = load_layers(CHECKPOINT)
        direct = forward(convs, fc, prepared[2:10:2, 1:7:2].reshape(-1, 110))
        numpy.testing.assert_allclose(reference, direct, atol=1e-4, rtol=1e-4)

    def test_raw_capture_is_read_and_labels_remapped_as_in_training(self):
        fake = self.dir / 'fakepkg'
        (fake / 'hypso').mkdir(parents=True)
        (fake / 'hypso/__init__.py').write_text(FAKE_HYPSO)
        raw = numpy.load(self.dir / 'cube.npy').transpose(1, 2, 0)
        (self.dir / 'cap-l1a.nc').write_bytes(b'')
        numpy.save(self.dir / 'cap-l1a.nc.cube.npy', raw)
        numpy.save(self.dir / 'cap-l1a_labels.npy', self.rng.integers(1, 4, (12, 9)).astype(numpy.uint8))
        (self.dir / 'cap-meta.json').write_text(json.dumps({'r_band': 56, 'g_band': 67, 'b_band': 86}))
        env = dict(os.environ, PYTHONPATH=str(fake))
        img, _, stdout = self.prepare('cap-l1a.nc', 'cap-l1a_labels.npy', env=env)
        raw_labels = numpy.load(self.dir / 'cap-l1a_labels.npy')[2:10:2, 1:7:2]
        numpy.testing.assert_array_equal(numpy.load(img / 'labels.npy'), raw_labels.astype(int) - 1)
        self.assertEqual(json.loads((img / 'meta.json').read_text())['rgb_bands'], [56, 67, 86])
        self.assertIn('remapped', stdout)

    def test_compare_before_the_board_step_says_what_to_run(self):
        img, _, _ = self.prepare()
        result = self.tool('compare', img)
        self.assertEqual(result.returncode, 1)
        self.assertIn('run the board step first', result.stderr)
        self.assertIn('notebook', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_cube_with_wrong_band_count_is_rejected(self):
        numpy.save(self.dir / 'bad.npy', numpy.zeros((4, 4, 100), dtype=numpy.float32))
        result = self.tool('prepare', self.dir / 'bad.npy', '--out', self.dir / 'x')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('120 raw bands', result.stderr)


@unittest.skipIf(numpy is None, 'exporter needs numpy')
@unittest.skipUnless(CHECKPOINT.exists(), 'checkpoint not present')
class ExportTests(unittest.TestCase):
    def test_committed_files_match_a_fresh_export(self):
        header = (ROOT / 'src/hls/justoliunet/justoliunet_weights.hpp').read_text()
        normalization = re.search(r'// Input normalization: (.*)', header).group(1)
        if normalization.startswith('PLACEHOLDER'):
            options = ['--placeholder-normalization']
        else:
            mu_sd = ROOT / normalization
            if not mu_sd.exists():
                self.skipTest(f'{normalization} not present')
            options = ['--mu-sd', str(mu_sd)]
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / 'tools/export_justoliunet.py'),
                            str(CHECKPOINT), *options, '--root', tmp],
                           check=True, capture_output=True, cwd=ROOT)
            for path in ('src/hls/justoliunet/justoliunet_weights.hpp',
                         'tb/justoliunet/justoliunet_vectors.hpp',
                         'tb/data/justoliunet_vectors.txt'):
                with self.subTest(path=path):
                    # Line endings as git stores them: a Windows checkout may add \r.
                    fresh = (Path(tmp) / path).read_bytes().replace(b'\r\n', b'\n')
                    committed = (ROOT / path).read_bytes().replace(b'\r\n', b'\n')
                    self.assertTrue(fresh == committed, f'{path} is stale; re-run tools/export_justoliunet.py')


if __name__ == '__main__':
    unittest.main()
