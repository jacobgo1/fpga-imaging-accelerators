"""justoliunet: board script against a simulated board, and generated files vs exporter.

The simulated board answers each spectrum with the logits recorded for it in
tb/data/justoliunet_vectors.txt, so this checks the float32 register traffic
and the comparison logic, not the network itself (native/csim do that).
"""
import filecmp
import json
import os
import re
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import tkinter
except ImportError:
    tkinter = None
try:
    import numpy
except ImportError:
    numpy = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_jtag', ROOT / 'tests/test_jtag.py')
jtag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jtag)

VECTORS = ROOT / 'tb/data/justoliunet_vectors.txt'
CHECKPOINT = ROOT / 'weights/fp32/justoliunet/justoliunet_pixel1d_e31_miou0-908_K110.pt'
SPECTRUM, LOGITS, BANDS = 0x400, 0x10, 120
HEADER = f"""\
#define XJUSTOLIUNET_CONTROL_ADDR_AP_CTRL       0x000
#define XJUSTOLIUNET_CONTROL_ADDR_LOGITS_BASE   0x{LOGITS:03x}
#define XJUSTOLIUNET_CONTROL_ADDR_LOGITS_HIGH   0x01f
#define XJUSTOLIUNET_CONTROL_WIDTH_LOGITS       32
#define XJUSTOLIUNET_CONTROL_DEPTH_LOGITS       3
#define XJUSTOLIUNET_CONTROL_ADDR_SPECTRUM_BASE 0x{SPECTRUM:03x}
#define XJUSTOLIUNET_CONTROL_ADDR_SPECTRUM_HIGH 0x5ff
#define XJUSTOLIUNET_CONTROL_WIDTH_SPECTRUM     32
#define XJUSTOLIUNET_CONTROL_DEPTH_SPECTRUM     {BANDS}
"""


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]


def recorded_vectors():
    table = {}
    for line in VECTORS.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        inputs, expected = line.split('|')
        table[tuple(f32_bits(v) for v in inputs.split())] = [float(v) for v in expected.split()]
    return table


class FakeClassifier(jtag.FakeBoard):
    def __init__(self, tcl, broken=False):
        super().__init__(tcl, broken)
        self.table = recorded_vectors()

    def run(self):
        self.starts += 1
        key = tuple(self.words.get(jtag.BASE + SPECTRUM + 4 * i, 0) for i in range(BANDS))
        logits = list(self.table.get(key, [0.0, 0.0, 0.0]))
        if self.broken:
            logits[1] += 0.01
        for i, v in enumerate(logits):
            self.words[jtag.BASE + LOGITS + 4 * i] = f32_bits(v)
        self.done = True


@unittest.skipIf(tkinter is None, 'Optional Tcl tests require Python tkinter; no display needed')
class JustoLiuNetBoardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        run = Path(self.tmp.name)
        (run / 'bitstream').mkdir()
        (run / 'bitstream/system_wrapper.bit').write_text('')
        (run / 'board').mkdir()
        (run / 'board/xjustoliunet_hw.h').write_text(HEADER)
        (run / 'board/address.tcl').write_text(f'set kernel_base 0x{jtag.BASE:08X}\n')
        (run / 'board/psu_init.tcl').write_text(
            'proc psu_init {} {}\nproc psu_ps_pl_isolation_removal {} {}\nproc psu_ps_pl_reset_config {} {}\n')
        self.run_dir = run

    def tearDown(self):
        self.tmp.cleanup()

    def open_board(self, broken=False):
        tcl = tkinter.Tcl()
        board = FakeClassifier(tcl, broken)
        tcl.eval('rename after _after; proc after {args} {}')
        tcl.eval('source ' + jtag.tcl_path(ROOT / 'software/jtag/justoliunet.tcl'))
        tcl.call('board_open', self.run_dir.as_posix())
        return tcl, board

    def test_float_bits_round_trip(self):
        tcl = tkinter.Tcl()
        tcl.eval('source ' + jtag.tcl_path(ROOT / 'software/jtag/justoliunet.tcl'))
        self.assertEqual(int(tcl.call('jl_bits', -1.5)), 0xBFC00000)
        self.assertEqual(float(tcl.call('jl_float', 0xBFC00000)), -1.5)
        self.assertEqual(int(tcl.call('jl_bits', 0.1)), f32_bits(0.1))

    def test_selftest_passes_when_fpga_matches_reference(self):
        tcl, board = self.open_board()
        self.assertEqual(int(tcl.call('jl_selftest')), 0)
        self.assertEqual(board.starts, len(recorded_vectors()))

    def test_selftest_catches_a_wrong_logit(self):
        tcl, _ = self.open_board(broken=True)
        self.assertEqual(int(tcl.call('jl_selftest')), 1)

    def test_classify_file_returns_the_class(self):
        tcl, _ = self.open_board()
        inputs, expected = self.first_vector()
        path = self.run_dir / 'pixel.txt'
        path.write_text('\n'.join(inputs))
        want = max(range(3), key=lambda c: float(expected[c]))
        self.assertEqual(int(tcl.call('jl_classify_file', path.as_posix())), want)

    def test_wrong_band_count_is_rejected(self):
        tcl, _ = self.open_board()
        with self.assertRaisesRegex(tkinter.TclError, '120 bands, got 3'):
            tcl.call('jl_classify', '0.1 0.2 0.3')

    def first_vector(self):
        line = next(l for l in VECTORS.read_text().splitlines() if not l.startswith('#'))
        inputs, expected = line.split('|')
        return inputs.split(), expected.split()

    def test_classify_pixels_writes_logits_in_order(self):
        tcl, board = self.open_board()
        lines = [l for l in VECTORS.read_text().splitlines() if not l.startswith('#')]
        pixels = self.run_dir / 'pixels.txt'
        pixels.write_text('# header\n' + '\n'.join(l.split('|')[0] for l in lines) + '\n')
        out = self.run_dir / 'logits.txt'
        tcl.call('jl_classify_pixels', pixels.as_posix(), out.as_posix())
        got = [[float(v) for v in l.split()] for l in out.read_text().splitlines()]
        want = [[float(v) for v in l.split('|')[1].split()] for l in lines]
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            for a, b in zip(g, w):
                self.assertAlmostEqual(a, b, delta=1e-6 + 1e-6 * abs(b))
        self.assertEqual(board.starts, len(lines))

    def test_pixels_option_classifies_a_file_and_exits(self):
        line = next(l for l in VECTORS.read_text().splitlines() if not l.startswith('#'))
        pixels = self.run_dir / 'pixels.txt'
        pixels.write_text(line.split('|')[0] + '\n')
        out = self.run_dir / 'logits.txt'
        tcl = tkinter.Tcl()
        FakeClassifier(tcl)
        tcl.eval('rename after _after; proc after {args} {}')
        tcl.eval('set exit_code -1; proc exit {code} {set ::exit_code $code; error "test exit"}')
        args = ' '.join(jtag.tcl_path(p) for p in (self.run_dir, Path('-pixels'), pixels, out))
        tcl.eval(f'set argv [list {args}]')
        with self.assertRaises(tkinter.TclError):
            tcl.eval('source ' + jtag.tcl_path(ROOT / 'software/jtag/justoliunet.tcl'))
        self.assertEqual(int(tcl.getvar('exit_code')), 0)
        self.assertEqual(len(out.read_text().splitlines()), 1)

    def test_script_has_complete_tcl_syntax(self):
        text = (ROOT / 'software/jtag/justoliunet.tcl').read_text()
        self.assertEqual(int(tkinter.Tcl().call('info', 'complete', text)), 1)


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
        for name in ('rgb', 'fpga_classes', 'reference_classes', 'mismatch', 'labels'):
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
                    self.assertTrue(filecmp.cmp(Path(tmp) / path, ROOT / path, shallow=False),
                                    f'{path} is stale; re-run tools/export_justoliunet.py')


if __name__ == '__main__':
    unittest.main()
