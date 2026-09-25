"""justoliunet: board script against a simulated board, and generated files vs exporter.

The simulated board answers each spectrum with the logits recorded for it in
tb/data/justoliunet_vectors.txt, so this checks the float32 register traffic
and the comparison logic, not the network itself (native/csim do that).
"""
import filecmp
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
SPECTRUM, LOGITS, BANDS = 0x400, 0x10, 110
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
        with self.assertRaisesRegex(tkinter.TclError, '110 bands, got 3'):
            tcl.call('jl_classify', '0.1 0.2 0.3')

    def first_vector(self):
        line = next(l for l in VECTORS.read_text().splitlines() if not l.startswith('#'))
        inputs, expected = line.split('|')
        return inputs.split(), expected.split()

    def test_script_has_complete_tcl_syntax(self):
        text = (ROOT / 'software/jtag/justoliunet.tcl').read_text()
        self.assertEqual(int(tkinter.Tcl().call('info', 'complete', text)), 1)


@unittest.skipIf(numpy is None, 'exporter needs numpy')
@unittest.skipUnless(CHECKPOINT.exists(), 'checkpoint not present')
class ExportTests(unittest.TestCase):
    def test_committed_files_match_a_fresh_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / 'tools/export_justoliunet.py'),
                            str(CHECKPOINT), '--root', tmp],
                           check=True, capture_output=True, cwd=ROOT)
            for path in ('src/hls/justoliunet/justoliunet_weights.hpp',
                         'tb/justoliunet/justoliunet_vectors.hpp',
                         'tb/data/justoliunet_vectors.txt'):
                with self.subTest(path=path):
                    self.assertTrue(filecmp.cmp(Path(tmp) / path, ROOT / path, shallow=False),
                                    f'{path} is stale; re-run tools/export_justoliunet.py')


if __name__ == '__main__':
    unittest.main()
