"""The PYNQ path: packaging a build for the board, the notebooks, fpga.sh.

The notebooks themselves run end to end against a fake board in test_notebooks.py; here
only that they are valid and what goes into the package next to them.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
HEADER = '#define XJUSTOUNETSIMPLE_CONTROL_ADDR_AP_CTRL 0x0\n'
# a golden kernel's package: the shared golden notebook, which model it is, and mu_sd.txt
PACKAGE = ['justounetsimple.bit', 'justounetsimple.hwh', 'xjustounetsimple_hw.h',
           'classify_golden.ipynb', 'kernel.json', 'mu_sd.txt']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flow = load_module('flow', ROOT / 'main.py')


def make_build(directory, kernel='justounetsimple', header=HEADER, hwh=True):
    """A bitstream run folder as main.py and Vivado leave it, with only the files packaging reads."""
    run = Path(directory) / f'build/{kernel}/2026-09-29_10-00-00-bitstream'
    impl = run / 'vivado/system.runs/impl_1'
    impl.mkdir(parents=True)
    (impl / 'system_wrapper.bit').write_bytes(b'BIT')
    if hwh:
        handoff = run / 'vivado/system.gen/sources_1/bd/system/hw_handoff'
        handoff.mkdir(parents=True)
        (handoff / 'system.hwh').write_bytes(b'<HWH/>')
    driver = run / f'hls/solution/impl/ip/drivers/{kernel}_v1_0/src'
    driver.mkdir(parents=True)
    (driver / f'x{kernel}_hw.h').write_text(header)
    return run


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def package(self, run, kernel):
        with contextlib.redirect_stdout(io.StringIO()):
            archive = flow.package_pynq(run, kernel)
        return run / f'{kernel}_pynq', archive

    def test_package_has_matching_names_and_everything_the_notebook_uses(self):
        out, archive = self.package(make_build(self.dir), 'justounetsimple')
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(PACKAGE))
        self.assertEqual((out / 'justounetsimple.bit').read_bytes(), b'BIT')
        self.assertEqual((out / 'justounetsimple.hwh').read_bytes(), b'<HWH/>')
        self.assertEqual((out / 'mu_sd.txt').read_bytes(), (ROOT / 'mu_sd.txt').read_bytes())
        with zipfile.ZipFile(archive) as zip_file:
            self.assertEqual(sorted(zip_file.namelist()), sorted(f'justounetsimple_pynq/{n}' for n in PACKAGE))

    def test_optimized_packages_take_the_shared_notebook_and_say_their_model(self):
        for kernel, model in (('justounetsimple_opt_zynq7020', 'justounetsimple'),
                              ('sp_unet_small_opt_direct', 'sp_unet_small'), ('justoliunet_bn_opt', 'justoliunet_bn')):
            with self.subTest(kernel=kernel):
                run = make_build(self.dir / kernel, kernel, f'#define X{kernel.upper()}_CONTROL_ADDR_AP_CTRL 0x0\n')
                out, _ = self.package(run, kernel)
                self.assertEqual(sorted(p.name for p in out.iterdir()),
                                 sorted([f'{kernel}.bit', f'{kernel}.hwh', f'x{kernel}_hw.h', 'classify.ipynb',
                                         'kernel.json']))
                self.assertEqual((out / 'classify.ipynb').read_bytes(),
                                 (ROOT / 'software/pynq/shared/classify.ipynb').read_bytes())
                info = json.loads((out / 'kernel.json').read_text())
                self.assertEqual(info, {'kernel': kernel, 'model': model, **flow.MODELS.MODELS[model]})

    def test_missing_hwh_is_named(self):
        with self.assertRaisesRegex(SystemExit, 'block design description'):
            self.package(make_build(self.dir, hwh=False), 'justounetsimple')

    def test_newest_bitstream_run_is_found(self):
        older = self.dir / 'build/justounetsimple/2026-09-25_10-00-00-bitstream/vivado/system.runs/impl_1'
        older.mkdir(parents=True)
        (older / 'system_wrapper.bit').write_bytes(b'')
        make_build(self.dir)
        (self.dir / 'build/justounetsimple/2026-09-30_10-00-00-csynth').mkdir()
        saved, flow.ROOT = flow.ROOT, self.dir
        try:
            names = [run.name for run in flow.bitstream_runs('justounetsimple')]
        finally:
            flow.ROOT = saved
        self.assertEqual(names, ['2026-09-25_10-00-00-bitstream', '2026-09-29_10-00-00-bitstream'])

    def test_every_notebook_is_valid_and_its_code_parses(self):
        import ast
        notebooks = sorted((ROOT / 'software/pynq').glob('*/*.ipynb'))
        self.assertEqual(len(notebooks), 2, 'the two shared notebooks')
        for path in notebooks:
            with self.subTest(notebook=path.relative_to(ROOT).as_posix()):
                notebook = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(notebook['nbformat'], 4)
                code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
                self.assertTrue(code)
                for cell in code:
                    if not cell.lstrip().startswith(('!', '%')):  # shell and magic lines are not Python
                        ast.parse(cell)

    def test_a_kernel_of_no_model_takes_its_own_notebook(self):
        run = make_build(self.dir, 'example', '#define XEXAMPLE_CONTROL_ADDR_AP_CTRL 0x0\n')
        own = self.dir / 'software/pynq/example'
        own.mkdir(parents=True)
        (own / 'example.ipynb').write_text('{}')
        (own / 'include.txt').write_text('mu_sd.txt    # a repository file it needs\n')
        (self.dir / 'mu_sd.txt').write_text('mu sd')
        saved, flow.ROOT = flow.ROOT, self.dir
        try:
            out, archive = self.package(run, 'example')
        finally:
            flow.ROOT = saved
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ['example.bit', 'example.hwh', 'example.ipynb', 'mu_sd.txt', 'xexample_hw.h'])
        with zipfile.ZipFile(archive) as zip_file:
            self.assertIn('example_pynq/example.ipynb', zip_file.namelist())


# A .bit as Vivado writes it: a header of fields, then 'e', the length and the words.
BIT_DATA = bytes.fromhex('ffffffff aa995566 20000000 30008001 0000000d')
BIT = (bytes.fromhex('0009 0ff00ff00ff00ff000 0001') + b'a' + (12).to_bytes(2, 'big') + b'system;Id=0\0'
       + b'b' + (12).to_bytes(2, 'big') + b'7z020clg484\0' + b'c' + (11).to_bytes(2, 'big') + b'2026/10/16\0'
       + b'd' + (9).to_bytes(2, 'big') + b'10:00:00\0' + b'e' + len(BIT_DATA).to_bytes(4, 'big') + BIT_DATA)
HWH = """<EDKSYSTEM><MODULES>
<MODULE INSTANCE="ps" MODTYPE="processing_system7"><PARAMETERS>
  <PARAMETER NAME="PCW_FPGA0_PERIPHERAL_FREQMHZ" VALUE="78.125"/>
  <PARAMETER NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ" VALUE="76.923080"/>
</PARAMETERS><MEMORYMAP>
  <MEMRANGE BASEVALUE="0x00000000" HIGHVALUE="0x3FFFFFFF" INSTANCE="ps" MEMTYPE="MEMORY"/>
  <MEMRANGE BASEVALUE="0x40000000" HIGHVALUE="0x4000FFFF" INSTANCE="kernel" MEMTYPE="REGISTER"
            MASTERBUSINTERFACE="M_AXI_GP0" SLAVEBUSINTERFACE="s_axi_control"/>
</MEMORYMAP></MODULE></MODULES></EDKSYSTEM>"""


# The 7020 kernel's register map as HLS writes it (offsets as in its build).
HEADER_7020 = """\
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_ADDR_AP_CTRL     0x00
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_ADDR_DIN0_DATA   0x10
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_BITS_DIN0_DATA   64
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_ADDR_DIN1_DATA   0x1c
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_BITS_DIN1_DATA   64
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_ADDR_DOUT_DATA   0x28
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_BITS_DOUT_DATA   64
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_ADDR_HEIGHT_DATA 0x34
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_BITS_HEIGHT_DATA 32
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_ADDR_WIDTH_DATA  0x3c
#define XJUSTOUNETSIMPLE_OPT_ZYNQ7020_CONTROL_BITS_WIDTH_DATA  32
"""


class OpuPackageTests(unittest.TestCase):
    """The zip for a board running the OPU system's PetaLinux (the ZedBoard)."""
    KERNEL = 'justounetsimple_opt_zynq7020'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = make_build(self.tmp.name, self.KERNEL, HEADER_7020)
        (self.run / 'vivado/system.runs/impl_1/system_wrapper.bit').write_bytes(BIT)
        (self.run / 'vivado/system.gen/sources_1/bd/system/hw_handoff/system.hwh').write_text(HWH)

    def tearDown(self):
        self.tmp.cleanup()

    def package(self, board=None, dtc=None, compiler=None):
        """Packs with only the tools given: dtc and the ARM compiler, found on PATH or not."""
        saved, saved_cc = flow.shutil.which, os.environ.pop('OPU_CC', None)
        tools = {'dtc': dtc, 'arm-linux-gnueabihf-gcc': compiler}
        flow.shutil.which = lambda name: tools.get(name, saved(name))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return flow.package(self.run, self.KERNEL, board)
        finally:
            flow.shutil.which = saved
            if saved_cc is not None:
                os.environ['OPU_CC'] = saved_cc

    def test_bit_to_bin_drops_the_header_and_swaps_each_word(self):
        data = flow.bit_to_bin(BIT)
        self.assertEqual(data, bytes.fromhex('ffffffff 665599aa 00000020 01800030 0d000000'))
        with self.assertRaisesRegex(SystemExit, 'Not a bitstream'):
            flow.bit_to_bin(b'BIT')
        with self.assertRaisesRegex(SystemExit, 'Not a bitstream'):
            flow.bit_to_bin(BIT[:-1])          # cut short

    def test_kernel_registers_and_clock_come_from_the_hwh(self):
        hwh = self.run / 'vivado/system.gen/sources_1/bd/system/hw_handoff/system.hwh'
        self.assertEqual(flow.hwh_kernel(hwh), (0x40000000, 0x10000, 76923080))
        hwh.write_text('<EDKSYSTEM/>')
        with self.assertRaisesRegex(SystemExit, 'no kernel registers'):
            flow.hwh_kernel(hwh)

    def test_zedboard_gets_what_fpgautil_loads(self):
        archive = self.package('zedboard')
        out = self.run / f'{self.KERNEL}_opu'
        names = [f'{self.KERNEL}.bit.bin', f'{self.KERNEL}.bit.bin.md5sum', f'{self.KERNEL}.dtsi',
                 f'x{self.KERNEL}_hw.h', 'load.sh', 'classify.c', 'kernel_regs.h']
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(names))
        data = (out / f'{self.KERNEL}.bit.bin').read_bytes()
        self.assertEqual(data, flow.bit_to_bin(BIT))
        # md5sum -c on the board: the hash, two spaces, where the file lies there
        self.assertEqual((out / f'{self.KERNEL}.bit.bin.md5sum').read_text(),
                         f'{hashlib.md5(data).hexdigest()}  /media/sd-pl/firmware/{self.KERNEL}.bit.bin\n')
        dtsi = (out / f'{self.KERNEL}.dtsi').read_text()
        for line in ('/plugin/;', 'target = <&fpga_full>;', f'firmware-name = "{self.KERNEL}.bit.bin";',
                     'target = <&amba>;', 'assigned-clock-rates = <76923080>;', 'kernel@40000000 {',
                     'reg = <0x40000000 0x10000>;', 'compatible = "generic-uio";'):
            self.assertIn(line, dtsi)
        self.assertEqual(dtsi.count('{'), dtsi.count('}'))
        load = (out / 'load.sh').read_text()
        for line in (f'md5sum -c {self.KERNEL}.bit.bin.md5sum', f'fpgautil -b {self.KERNEL}.bit.bin -o {self.KERNEL}.dtbo',
                     'devmem 0x40000000', 'built for 76923 kHz', '-gt 76923 ]',
                     'divider=$(( (33333 * ((io >> 12) & 0x7f) + 76923 - 1) / 76923 ))',
                     'devmem 0xF8000170 32 $(( (fclk & ~0x03F03F30) | (divider << 8) | (1 << 20) ))'):
            self.assertIn(line, load)
        self.assertNotIn('\r', load)
        with zipfile.ZipFile(archive) as zip_file:
            self.assertEqual(sorted(zip_file.namelist()), sorted(f'{self.KERNEL}_opu/{n}' for n in names))

    def test_dtc_compiles_the_overlay_when_there_is_one(self):
        calls = []
        saved = flow.subprocess.run
        flow.subprocess.run = lambda command, **kw: calls.append(command)
        try:
            self.package('zedboard', dtc='/usr/bin/dtc')
        finally:
            flow.subprocess.run = saved
        out = self.run / f'{self.KERNEL}_opu'
        self.assertEqual(calls, [['/usr/bin/dtc', '-@', '-q', '-I', 'dts', '-O', 'dtb',
                                  '-o', str(out / f'{self.KERNEL}.dtbo'), str(out / f'{self.KERNEL}.dtsi')]])

    def test_classify_is_built_for_the_board_with_the_kernels_registers(self):
        calls = []
        saved = flow.subprocess.run
        flow.subprocess.run = lambda command, **kw: calls.append(command)
        try:
            self.package('zedboard', compiler='/usr/bin/arm-linux-gnueabihf-gcc')
        finally:
            flow.subprocess.run = saved
        out = self.run / f'{self.KERNEL}_opu'
        source = ROOT / 'software/opu/justounetsimple_opt/classify.c'    # the 7020's include.txt
        self.assertEqual(calls, [['/usr/bin/arm-linux-gnueabihf-gcc', '-O2', '-Wall', '-static', f'-I{out}',
                                  str(source), '-o', str(out / 'classify')]])
        self.assertEqual((out / 'classify.c').read_bytes(), source.read_bytes())

    @unittest.skipIf(BASH is None, 'load.sh needs a shell')
    def test_load_sh_sets_the_clock_when_the_overlay_did_not(self):
        """load.sh with stand-ins for the board's commands: devmem reads the PS registers
        from a file and records writes. FCLK_CLK0 at 100 MHz (IO PLL 1 GHz / 10) must go to
        1 GHz / 13 = 76.9 MHz, the fastest not above the kernel's 76.92; at that it stays."""
        self.package('zedboard')
        out = self.run / f'{self.KERNEL}_opu'
        bin_dir = Path(self.tmp.name) / 'bin'
        bin_dir.mkdir()
        stubs = {
            'devmem': '#!/bin/sh\n'
                      'if [ $# -eq 1 ]; then\n'
                      '  case $(( $1 )) in\n'
                      '    $((0xF8000108))) echo 0x0001E000 ;;\n'        # IO PLL: FDIV 30
                      '    $((0xF8000170))) cat "$STATE" ;;\n'
                      '    *) echo 0x00000004 ;;\n'
                      '  esac\n'
                      'else\n'
                      '  echo "$1 $3" >> "$STATE.writes"\n'
                      '  if [ $(( $1 )) -eq $((0xF8000170)) ]; then printf "0x%08X\\n" $(( $3 )) > "$STATE"; fi\n'
                      'fi\n',
            'fpgautil': '#!/bin/sh\necho "fpgautil $*" >> "$STATE.writes"\n',
            'md5sum': '#!/bin/sh\nexit 0\n',
            'dtc': '#!/bin/sh\nexit 0\n',
        }
        for name, text in stubs.items():
            (bin_dir / name).write_text(text, newline='\n')
        load = (out / 'load.sh').read_text().replace('cd /media/sd-pl/firmware', 'cd "$PACKAGE"')
        (out / 'load_here.sh').write_text(load, newline='\n')
        state = Path(self.tmp.name) / 'fclk'
        for start, want, writes in (('0x00100A00', '0x00100D00', 2), ('0x00100D00', '0x00100D00', 0)):
            with self.subTest(start=start):
                state.write_text(start + '\n')
                Path(f'{state}.writes').write_text('')
                env = {**os.environ, 'STATE': state.as_posix(), 'PACKAGE': out.as_posix(),
                       'PATH': bin_dir.as_posix() + os.pathsep + os.environ['PATH']}
                result = subprocess.run([BASH, '-c', f'PATH="{bin_dir.as_posix()}:$PATH" sh load_here.sh'],
                                        cwd=out, capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(state.read_text().strip(), want)
                logged = Path(f'{state}.writes').read_text().splitlines()
                self.assertEqual(len([w for w in logged if w.startswith('0xF8')]), writes, logged)
                self.assertIn(f'fpgautil -b {self.KERNEL}.bit.bin -o {self.KERNEL}.dtbo', logged)
                self.assertIn('FCLK_CLK0: 76922 kHz, built for 76923 kHz', result.stdout)

    def test_register_offsets_for_the_programs(self):
        regs = flow.opu_registers(HEADER_7020)
        for line in ('#define REG_AP_CTRL 0x00', '#define REG_HEIGHT 0x34', '#define REG_WIDTH 0x3c',
                     '#define REG_DINS {0x10, 0x1c}', '#define REG_DIN_64 1', '#define REG_DOUT 0x28',
                     '#define REG_DOUT_64 1'):
            self.assertTrue(any(l == line or l.startswith(line + ' ') for l in regs.splitlines()), line)
        narrow = HEADER_7020.replace('BITS_DIN0_DATA   64', 'BITS_DIN0_DATA   32')
        self.assertIn('#define REG_DIN_64 0', flow.opu_registers(narrow))
        with self.assertRaisesRegex(SystemExit, 'WIDTH_DATA'):
            flow.opu_registers(HEADER_7020.replace('ADDR_WIDTH_DATA', 'ADDR_OTHER'))

    def test_the_run_says_which_board_it_was_for(self):
        (self.run / 'run.json').write_text(json.dumps({'board': 'zedboard'}))
        self.assertTrue(self.package().name.endswith('_opu.zip'))
        (self.run / 'run.json').write_text(json.dumps({'board': 'zynq7030'}))   # PicoBoB, the OPU system
        self.assertTrue(self.package().name.endswith('_opu.zip'))
        (self.run / 'run.json').write_text(json.dumps({'board': 'zynq7020'}))
        self.assertTrue(self.package().name.endswith('_pynq.zip'))
        (self.run / 'run.json').unlink()                  # the default board's
        self.assertTrue(self.package().name.endswith('_pynq.zip'))


@unittest.skipIf(BASH is None, 'fpga.sh needs bash (Git Bash, WSL or Linux)')
class FpgaShellTests(unittest.TestCase):
    def bash(self, script, root=ROOT, cwd=None):
        prefix = (f'export FPGA_ROOT="{Path(root).as_posix()}"\n'
                  f'export FPGA_PYTHON="{Path(sys.executable).as_posix()}"\n'  # python3 may be a stub on Windows
                  f'source "{(ROOT / "fpga.sh").as_posix()}"\n')
        return subprocess.run([BASH, '-c', prefix + script], text=True, capture_output=True, cwd=cwd)

    def complete(self, *words, root=ROOT):
        """What Tab offers after `words` (the last one is the word being completed)."""
        array = ' '.join(f'"{w}"' for w in words)
        result = self.bash(f'COMP_WORDS=({array}); COMP_CWORD={len(words) - 1}; _fpga_complete; '
                           'printf "%s\\n" "${COMPREPLY[@]}"', root)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.split()

    def test_help_lists_every_command(self):
        result = self.bash('fpga_help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ('fpga_kernels', 'fpga_env', 'fpga_test', 'fpga_build', 'fpga_ip', 'fpga_cosim',
                        'fpga_runs', 'fpga_pynq', 'fpga_zip'):
            self.assertIn(command, result.stdout)

    def test_kernels_are_the_folders_in_src_hls(self):
        result = self.bash('fpga_kernels')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), sorted(p.name for p in (ROOT / 'src/hls').iterdir() if p.is_dir()))

    def test_tab_completes_kernel_names(self):
        for command in ('fpga_test', 'fpga_build', 'fpga_ip', 'fpga_cosim', 'fpga_runs', 'fpga_pynq'):
            with self.subTest(command=command):
                self.assertEqual(self.complete(command, 'justounetsimple'),
                                 ['justounetsimple', 'justounetsimple_opt', 'justounetsimple_opt_zynq7020',
                                  'justounetsimple_opt_zynq7030'])
        self.assertEqual(self.complete('fpga_build', 'justounetsimple', '--c'), ['--clock-ns'])
        self.assertEqual(self.complete('fpga_cosim', 'justounetsimple', '--board', 'zynq7'), ['zynq7020', 'zynq7030'])
        self.assertEqual(self.complete('fpga_cosim', 'justounetsimple', '--d'), ['--dataflow-profile'])
        self.assertEqual(self.complete('fpga_build', 'justounetsimple', '--d'), [])
        self.assertEqual(self.complete('fpga_build', 'justounetsimple', '--board', ''), ['zcu104', 'zedboard', 'zynq7020', 'zynq7030'])
        self.assertEqual(self.complete('fpga_build', 'justounetsimple', '--board', 'zynq7030', '--'),
                         ['--board', '--clock-ns'])
        self.assertEqual(self.complete('fpga_ip', 'justounetsimple', '--board', 'zed'), ['zedboard'])
        self.assertEqual(self.complete('fpga_ip', 'justounetsimple', '--'), ['--board', '--clock-ns'])

    def test_tab_completes_a_kernels_bitstream_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_build(tmp)
            (Path(tmp) / 'build/justounetsimple/2026-09-30_10-00-00-csynth').mkdir()
            self.assertEqual(self.complete('fpga_pynq', 'justounetsimple', '', root=tmp),
                             ['2026-09-29_10-00-00-bitstream'])

    def test_completion_is_registered(self):
        result = self.bash('complete -p fpga_build fpga_ip fpga_cosim fpga_pynq fpga_zip')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('complete -F _fpga_complete fpga_build', result.stdout)
        self.assertIn('complete -F _fpga_complete fpga_ip', result.stdout)
        self.assertIn('complete -F _fpga_complete fpga_cosim', result.stdout)
        self.assertIn('complete -d fpga_zip', result.stdout)

    def test_runs_lists_bitstream_runs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_build(tmp)
            (Path(tmp) / 'build/justounetsimple/2026-09-30_10-00-00-csynth').mkdir()
            result = self.bash('fpga_runs justounetsimple', tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([Path(line).name for line in result.stdout.split()],
                             ['2026-09-29_10-00-00-bitstream'])

    def test_kernel_must_be_named(self):
        for command in ('fpga_test', 'fpga_build', 'fpga_ip', 'fpga_cosim', 'fpga_runs', 'fpga_pynq'):
            with self.subTest(command=command):
                result = self.bash(command)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('usage', result.stderr)

    def test_zip_keeps_the_folder_name_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'aegean').mkdir()
            (Path(tmp) / 'aegean/cube.npy').write_bytes(b'')
            result = self.bash('fpga_zip aegean', cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            with zipfile.ZipFile(Path(tmp) / 'aegean.zip') as zip_file:
                self.assertIn('aegean/cube.npy', zip_file.namelist())


if __name__ == '__main__':
    unittest.main()
