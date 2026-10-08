"""The PYNQ path: packaging a build for the board, the notebooks, fpga.sh.

The notebooks themselves run end to end against a fake board in test_justounetsimple.py;
here only that they are valid and what goes into the package next to them.
"""
import contextlib
import importlib.util
import io
import json
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
# justounetsimple's notebook, plus the repository file its include.txt names
PACKAGE = ['justounetsimple.bit', 'justounetsimple.hwh', 'xjustounetsimple_hw.h',
           'justounetsimple.ipynb', 'mu_sd.txt']


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
        self.assertTrue(notebooks)
        for path in notebooks:
            with self.subTest(notebook=path.relative_to(ROOT).as_posix()):
                notebook = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(notebook['nbformat'], 4)
                code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
                self.assertTrue(code)
                for cell in code:
                    if not cell.lstrip().startswith(('!', '%')):  # shell and magic lines are not Python
                        ast.parse(cell)

    def test_matmul_notebook_lands_next_to_its_bitstream(self):
        run = make_build(self.dir, 'matmul', '#define XMATMUL_CONTROL_ADDR_AP_CTRL 0x0\n')
        out, archive = self.package(run, 'matmul')
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ['matmul.bit', 'matmul.hwh', 'matmul.ipynb', 'xmatmul_hw.h'])
        self.assertEqual((out / 'matmul.ipynb').read_bytes(),
                         (ROOT / 'software/pynq/matmul/matmul.ipynb').read_bytes(), 'copied as is')
        with zipfile.ZipFile(archive) as zip_file:
            self.assertIn('matmul_pynq/matmul.ipynb', zip_file.namelist())


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
        for command in ('fpga_kernels', 'fpga_env', 'fpga_test', 'fpga_build', 'fpga_runs', 'fpga_pynq',
                        'fpga_zip'):
            self.assertIn(command, result.stdout)

    def test_kernels_are_the_folders_in_src_hls(self):
        result = self.bash('fpga_kernels')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), sorted(p.name for p in (ROOT / 'src/hls').iterdir() if p.is_dir()))

    def test_tab_completes_kernel_names(self):
        for command in ('fpga_test', 'fpga_build', 'fpga_runs', 'fpga_pynq'):
            with self.subTest(command=command):
                self.assertEqual(self.complete(command, 'justounetsimple'),
                                 ['justounetsimple', 'justounetsimple_opt'])
        self.assertEqual(self.complete('fpga_build', 'matmul', '--c'), ['--clock-ns'])
        self.assertEqual(self.complete('fpga_build', 'matmul', '--board', ''), ['zcu104', 'zynq7030'])
        self.assertEqual(self.complete('fpga_build', 'matmul', '--board', 'zynq7030', '--'),
                         ['--board', '--clock-ns'])

    def test_tab_completes_a_kernels_bitstream_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_build(tmp)
            (Path(tmp) / 'build/justounetsimple/2026-09-30_10-00-00-csynth').mkdir()
            self.assertEqual(self.complete('fpga_pynq', 'justounetsimple', '', root=tmp),
                             ['2026-09-29_10-00-00-bitstream'])

    def test_completion_is_registered(self):
        result = self.bash('complete -p fpga_build fpga_pynq fpga_zip')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('complete -F _fpga_complete fpga_build', result.stdout)
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
        for command in ('fpga_test', 'fpga_build', 'fpga_runs', 'fpga_pynq'):
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
