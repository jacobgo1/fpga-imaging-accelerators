"""Run software/jtag against a simulated board instead of xsdb.

The fake implements the HLS s_axilite array layout independently (16-bit
elements two per word, 64-bit elements over two words, low part first), so the
Tcl packing is checked against a model of the hardware, not against itself.
Nothing here proves the real header or board behave this way.
"""
from pathlib import Path
import tempfile
import unittest

try:
    import tkinter
except ImportError:
    tkinter = None

ROOT = Path(__file__).resolve().parents[1]
BASE = 0xA0000000
HEADER = """\
// control
// 0x000 : Control signals
#define XMATMUL_CONTROL_ADDR_AP_CTRL 0x000
#define XMATMUL_CONTROL_ADDR_GIE     0x004
#define XMATMUL_CONTROL_ADDR_IER     0x008
#define XMATMUL_CONTROL_ADDR_ISR     0x00c
#define XMATMUL_CONTROL_ADDR_A_BASE  0x080
#define XMATMUL_CONTROL_ADDR_A_HIGH  0x0ff
#define XMATMUL_CONTROL_WIDTH_A      16
#define XMATMUL_CONTROL_DEPTH_A      64
#define XMATMUL_CONTROL_ADDR_B_BASE  0x100
#define XMATMUL_CONTROL_ADDR_B_HIGH  0x17f
#define XMATMUL_CONTROL_WIDTH_B      16
#define XMATMUL_CONTROL_DEPTH_B      64
#define XMATMUL_CONTROL_ADDR_C_BASE  0x200
#define XMATMUL_CONTROL_ADDR_C_HIGH  0x3ff
#define XMATMUL_CONTROL_WIDTH_C      64
#define XMATMUL_CONTROL_DEPTH_C      64
"""


def signed(value, bits):
    return value - (1 << bits) if value >> (bits - 1) else value


class FakeBoard:
    def __init__(self, tcl, broken=False):
        self.tcl, self.broken = tcl, broken
        self.words, self.calls, self.done, self.starts = {}, [], False, 0
        for name in ('connect', 'targets', 'rst', 'fpga', 'mwr', 'mrd'):
            tcl.createcommand(name, lambda *args, name=name: self.command(name, args))

    def command(self, name, args):
        self.calls.append(' '.join((name,) + args))
        positional = [a for a in args if not a.startswith('-')]
        if name == 'mwr':
            address = int(positional[0], 0)
            for i, word in enumerate(self.tcl.splitlist(positional[1])):
                self.write(address + 4 * i, int(word, 0))
        if name == 'mrd':
            address = int(positional[0], 0)
            count = int(positional[1]) if len(positional) > 1 else 1
            return ' '.join(str(self.read(address + 4 * i)) for i in range(count))
        return ''

    def write(self, address, word):
        if address == BASE and word & 1:
            self.run()
        else:
            self.words[address] = word

    def read(self, address):
        if address == BASE:
            done, self.done = self.done, False
            return (int(done) << 1) | (int(not done) << 2)
        return self.words.get(address, 0)

    def int16s(self, offset):
        values = []
        for i in range(64):
            word = self.words.get(BASE + offset + 4 * (i // 2), 0)
            values.append(signed((word >> (16 * (i % 2))) & 0xFFFF, 16))
        return values

    def run(self):
        self.starts += 1
        a, b = self.int16s(0x080), self.int16s(0x100)
        for i in range(8):
            for j in range(8):
                total = sum(a[i * 8 + k] * b[k * 8 + j] for k in range(8))
                if self.broken and (i, j) == (3, 5):
                    total += 1
                raw = total & (2 ** 64 - 1)
                address = BASE + 0x200 + 8 * (i * 8 + j)
                self.words[address] = raw & 0xFFFFFFFF
                self.words[address + 4] = raw >> 32
        self.done = True


@unittest.skipIf(tkinter is None, 'Optional Tcl tests require Python tkinter; no display needed')
class JtagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        (self.run_dir / 'bitstream').mkdir()
        (self.run_dir / 'bitstream/system_wrapper.bit').write_text('')
        board = self.run_dir / 'board'
        board.mkdir()
        (board / 'xmatmul_hw.h').write_text(HEADER)
        (board / 'address.tcl').write_text(f'set kernel_base 0x00{BASE:08X}\n')
        (board / 'psu_init.tcl').write_text(
            'set psu_log {}\n'
            'proc psu_init {} {lappend ::psu_log psu_init}\n'
            'proc psu_ps_pl_isolation_removal {} {lappend ::psu_log isolation}\n'
            'proc psu_ps_pl_reset_config {} {lappend ::psu_log reset}\n')

    def tearDown(self):
        self.tmp.cleanup()

    def open_board(self, broken=False, *extra):
        tcl = tkinter.Tcl()
        board = FakeBoard(tcl, broken)
        tcl.eval('rename after _after; proc after {args} {}')
        tcl.eval('source ' + tcl_path(ROOT / 'software/jtag/matmul.tcl'))
        tcl.call('board_open', self.run_dir.as_posix(), *extra)
        return tcl, board

    def test_selftest_passes_on_a_correct_kernel(self):
        tcl, _ = self.open_board()
        self.assertEqual(int(tcl.call('mm_selftest', 5)), 0)

    def test_selftest_catches_a_wrong_result(self):
        tcl, _ = self.open_board(broken=True)
        self.assertEqual(int(tcl.call('mm_selftest', 1)), 1)

    def test_multiply_returns_signed_rows(self):
        tcl, _ = self.open_board()
        result = tcl.eval('mm_multiply [mm_constant -32768] [mm_constant 32767]')
        rows = [tcl.splitlist(r) for r in tcl.splitlist(result)]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(int(v) == -32768 * 32767 * 8 for row in rows for v in row))

    def test_pl_is_programmed_before_ps_init(self):
        tcl, board = self.open_board()
        names = [c.split()[0] for c in board.calls]
        self.assertLess(names.index('rst'), names.index('fpga'))
        self.assertEqual(tcl.eval('join $psu_log ,'), 'psu_init,isolation,reset')

    def test_explicit_base_replaces_missing_address_file(self):
        (self.run_dir / 'board/address.tcl').unlink()
        tcl, _ = self.open_board(False, hex(BASE))
        self.assertEqual(int(tcl.call('mm_selftest', 1)), 0)

    def test_header_with_too_small_window_is_rejected(self):
        header = self.run_dir / 'board/xmatmul_hw.h'
        header.write_text(HEADER.replace('ADDR_A_HIGH  0x0ff', 'ADDR_A_HIGH  0x0bf'))
        tcl = tkinter.Tcl()
        tcl.eval('source ' + tcl_path(ROOT / 'software/jtag/board.tcl'))
        with self.assertRaisesRegex(tkinter.TclError, 'packing assumption'):
            tcl.call('board_load_map', self.run_dir.as_posix())

    def live_session(self, text, broken=False):
        tcl, board = self.open_board(broken)
        session = self.run_dir / 'session.txt'
        session.write_text(text)
        failed = int(tcl.eval(f'set ch [open {tcl_path(session)}]; set r [mm_live $ch]; close $ch; set r'))
        return failed, board.starts

    def test_live_mode_runs_every_way_of_entering_a_matrix(self):
        matrix_file = self.run_dir / 'b.txt'
        matrix_file.write_text('\n'.join(', '.join(str(r * 8 + c - 32) for c in range(8)) for r in range(8)))
        typed_rows = '\n'.join(' '.join(str((r + 1) * (c - 3)) for c in range(8)) for r in range(8))
        session = [
            'random', 'identity',                     # run 1
            typed_rows, 'same',                       # run 2: typed A, previous B
            '1 2 x', '40000 ' + '0 ' * 63,            # rejected entries, re-prompted
            'random -1000 1000',
            f'file {matrix_file.as_posix()}',         # run 3
            'quit',
        ]
        self.assertEqual(self.live_session('\n'.join(session) + '\n'), (0, 3))

    def test_live_mode_reports_a_wrong_result(self):
        self.assertEqual(self.live_session('random\nrandom\nquit\n', broken=True), (1, 1))

    def test_live_mode_ends_cleanly_at_end_of_input(self):
        self.assertEqual(self.live_session('random\n'), (0, 0))

    def test_scripts_have_complete_tcl_syntax(self):
        tcl = tkinter.Tcl()
        for path in ('software/jtag/board.tcl', 'software/jtag/matmul.tcl', 'boards/zcu104/system.tcl'):
            self.assertEqual(int(tcl.call('info', 'complete', (ROOT / path).read_text())), 1, path)


def tcl_path(path):
    return '{' + path.as_posix() + '}'


if __name__ == '__main__':
    unittest.main()
