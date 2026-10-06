"""The PYNQ path: packaging a build, the notebook driver, the notebook, fpga.sh.

The driver runs against a simulated overlay whose kernel computes each pixel
with the numpy reference of the committed kernel, so this checks the register
traffic and the image handling -- not PYNQ or the board.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
SPECTRUM, LOGITS = 0x400, 0x10
HEADER = f"""\
#define XJUSTOLIUNET_CONTROL_ADDR_AP_CTRL       0x000
#define XJUSTOLIUNET_CONTROL_ADDR_LOGITS_BASE   0x{LOGITS:03x}
#define XJUSTOLIUNET_CONTROL_ADDR_LOGITS_HIGH   0x01f
#define XJUSTOLIUNET_CONTROL_WIDTH_LOGITS       32
#define XJUSTOLIUNET_CONTROL_DEPTH_LOGITS       3
#define XJUSTOLIUNET_CONTROL_ADDR_SPECTRUM_BASE 0x{SPECTRUM:03x}
#define XJUSTOLIUNET_CONTROL_ADDR_SPECTRUM_HIGH 0x5ff
#define XJUSTOLIUNET_CONTROL_WIDTH_SPECTRUM     32
#define XJUSTOLIUNET_CONTROL_DEPTH_SPECTRUM     120
"""
PACKAGE = ['justoliunet.bit', 'justoliunet.hwh', 'xjustoliunet_hw.h', 'justoliunet_pynq.py',
           'justoliunet.ipynb', 'justoliunet_vectors.txt', 'justoliunet_image.py',
           'export_justoliunet.py', 'pt_reader.py']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flow = load_module('flow', ROOT / 'main.py')


def make_build(directory, kernel='justoliunet', header=HEADER, hwh=True):
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
        out, archive = self.package(make_build(self.dir), 'justoliunet')
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(PACKAGE))
        self.assertEqual((out / 'justoliunet.bit').read_bytes(), b'BIT')
        self.assertEqual((out / 'justoliunet.hwh').read_bytes(), b'<HWH/>')
        with zipfile.ZipFile(archive) as zip_file:
            self.assertEqual(sorted(zip_file.namelist()), sorted(f'justoliunet_pynq/{n}' for n in PACKAGE))

    def test_missing_hwh_is_named(self):
        with self.assertRaisesRegex(SystemExit, 'block design description'):
            self.package(make_build(self.dir, hwh=False), 'justoliunet')

    def test_newest_bitstream_run_is_found(self):
        older = self.dir / 'build/justoliunet/2026-09-25_10-00-00-bitstream/vivado/system.runs/impl_1'
        older.mkdir(parents=True)
        (older / 'system_wrapper.bit').write_bytes(b'')
        make_build(self.dir)
        (self.dir / 'build/justoliunet/2026-09-30_10-00-00-csynth').mkdir()
        saved, flow.ROOT = flow.ROOT, self.dir
        try:
            names = [run.name for run in flow.bitstream_runs('justoliunet')]
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


class FakeRegisters:
    """A kernel's AXI-Lite registers, computing like the numpy reference."""

    def __init__(self, model, broken=False):
        self.model, self.broken = model, broken
        self.words, self.done, self.starts = {}, False, 0

    def read(self, offset):
        if offset == 0:
            done, self.done = self.done, False
            return (0x2 if done else 0) | 0x4
        return self.words.get(offset, 0)

    def write(self, offset, value):
        if offset == 0 and value & 1:
            self.starts += 1
            raw = np.array([self.words.get(SPECTRUM + 4 * i, 0) for i in range(120)], dtype=np.uint32)
            logits = self.model(raw.view(np.float32)).astype(np.float32)
            if self.broken:
                logits[1] += 0.5
            for i, bits in enumerate(logits.view(np.uint32)):
                self.words[LOGITS + 4 * i] = int(bits)
            self.done = True
        else:
            self.words[offset] = value


class FakeOverlay:
    """What the driver uses of pynq.Overlay: ip_dict and the IP's mmio."""

    def __init__(self, regs, name='kernel'):
        self.ip_dict = {'zynq_ultra_ps_e_0': {'type': 'xilinx.com:ip:zynq_ultra_ps_e:3.5'},
                        name: {'type': 'xilinx.com:hls:justoliunet:1.0'}}
        setattr(self, name, type('IP', (), {'mmio': regs})())


@unittest.skipIf(np is None, 'the driver needs numpy')
class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.driver = load_module('justoliunet_pynq', ROOT / 'software/pynq/justoliunet/justoliunet_pynq.py')
        export = load_module('export_justoliunet', ROOT / 'tools/export_justoliunet.py')
        header = (ROOT / export.WEIGHTS).read_text()
        checkpoint = ROOT / re.search(r'Generated by tools/export_justoliunet.py from (\S+)', header).group(1)
        if not checkpoint.exists():
            raise unittest.SkipTest('checkpoint not present')
        convs, fc = export.load_layers(checkpoint)
        _, kept, mean, inv_std, _ = export.read_preprocessing()
        cls.model = staticmethod(lambda raw: export.forward(
            convs, fc, export.preprocess(raw[None, :], kept, mean, inv_std))[0])

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / 'xjustoliunet_hw.h').write_text(HEADER)

    def tearDown(self):
        self.tmp.cleanup()

    def driver_for(self, broken=False, name='kernel'):
        self.regs = FakeRegisters(self.model, broken)
        return self.driver.JustoLiuNet(self.dir, overlay=FakeOverlay(self.regs, name))

    def test_selftest_passes_on_a_correct_kernel_and_fails_on_a_wrong_one(self):
        vectors = ROOT / 'tb/data/justoliunet_vectors.txt'
        self.assertTrue(self.driver_for().selftest(vectors))
        self.assertFalse(self.driver_for(broken=True).selftest(vectors))

    def test_kernel_is_found_by_its_type_whatever_the_block_is_called(self):
        jl = self.driver_for(name='justoliunet_0')
        self.assertEqual(jl.status(), {'start': False, 'done': False, 'idle': True})

    def test_wrong_band_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '120 bands'):
            self.driver_for().classify(np.zeros(110))

    def test_zipped_image_is_classified_and_matches_its_reference(self):
        cube = np.random.default_rng(0).uniform(100, 4000, (6, 5, 120)).astype(np.float32)
        np.save(self.dir / 'img.npy', cube)
        prepared = subprocess.run([sys.executable, str(ROOT / 'tools/justoliunet_image.py'), 'prepare',
                                   str(self.dir / 'img.npy'), '--out', str(self.dir / 'img')],
                                  capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        shutil.make_archive(str(self.dir / 'upload'), 'zip', self.dir, 'img')
        shutil.rmtree(self.dir / 'img')
        folder, logits = self.driver_for().classify_folder(self.dir / 'upload.zip')
        self.assertEqual(folder, self.dir / 'img')
        self.assertEqual(logits.shape, (30, 3))
        compared = subprocess.run([sys.executable, str(ROOT / 'tools/justoliunet_image.py'), 'compare',
                                   str(folder)], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(compared.returncode, 0, compared.stdout + compared.stderr)
        self.assertIn('MATCH', compared.stdout)


@unittest.skipIf(BASH is None, 'fpga.sh needs bash (Git Bash, WSL or Linux)')
class FpgaShellTests(unittest.TestCase):
    def bash(self, script, root=ROOT, cwd=None):
        prefix = (f'export FPGA_ROOT="{Path(root).as_posix()}"\n'
                  f'export FPGA_PYTHON="{Path(sys.executable).as_posix()}"\n'  # python3 may be a stub on Windows
                  f'source "{(ROOT / "fpga.sh").as_posix()}"\n')
        return subprocess.run([BASH, '-c', prefix + script], text=True, capture_output=True, cwd=cwd)

    def test_help_lists_every_command(self):
        result = self.bash('fpga_help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ('fpga_env', 'fpga_test', 'fpga_build', 'fpga_runs', 'fpga_pynq', 'fpga_zip'):
            self.assertIn(command, result.stdout)

    def test_runs_lists_bitstream_runs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_build(tmp)
            (Path(tmp) / 'build/justoliunet/2026-09-30_10-00-00-csynth').mkdir()
            result = self.bash('fpga_runs justoliunet', tmp)
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
            (Path(tmp) / 'aegean/pixels.txt').write_text('# 0 pixels\n')
            result = self.bash('fpga_zip aegean', cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            with zipfile.ZipFile(Path(tmp) / 'aegean.zip') as zip_file:
                self.assertIn('aegean/pixels.txt', zip_file.namelist())


if __name__ == '__main__':
    unittest.main()
