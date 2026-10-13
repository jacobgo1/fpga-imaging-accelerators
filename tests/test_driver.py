"""software/driver: the board's C++ driver for justounetsimple_opt, on the PC.

jopt_driver_sim is the driver with a kernel's own C++ in place of the FPGA
(software/driver/src/sim.cpp): it checks the register map, the strips, the
copies and the files against classify_fixed -- not the board's UIO or DDR.
Once for each kernel the driver serves: justounetsimple_opt (ZCU104) and, where
it exists, justounetsimple_opt_zynq7030 (HYPSO-2).
"""
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
DRIVER = ROOT / 'software/driver'
CXX = shutil.which('g++') or shutil.which('clang++')


def register_header(kernel, ports):
    """The layout Vitis HLS gives the kernel's registers (the driver reads the offsets)."""
    prefix = f'X{kernel.upper()}_CONTROL'
    lines = [f'#define {prefix}_ADDR_AP_CTRL 0x00', f'#define {prefix}_ADDR_GIE 0x04',
             f'#define {prefix}_ADDR_IER 0x08', f'#define {prefix}_ADDR_ISR 0x0c']
    offset = 0x10
    for name, bits in [*[(f'DIN{i}', 64) for i in range(ports)], ('DOUT', 64), ('HEIGHT', 32), ('WIDTH', 32)]:
        lines += [f'#define {prefix}_ADDR_{name}_DATA 0x{offset:02x}', f'#define {prefix}_BITS_{name}_DATA {bits}']
        offset += 12 if bits == 64 else 8
    return '\n'.join(lines) + '\n'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DriverTests:
    """The tests; the subclasses below say which kernel plays the FPGA."""
    KERNEL, PORTS, DEFINES = None, None, []
    H, W = 70, 45      # 3 x 2 patches, the edge ones partial

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('justounetsimple_model', ROOT / 'tools/justounetsimple_model.py')
        # The same files as CMakeLists.txt's jopt_driver_sim.
        cls.exe = cls.dir / 'jopt_driver_sim'
        kernel_dir = ROOT / 'src/hls' / cls.KERNEL
        sources = [*sorted((DRIVER / 'src').glob('*.cpp')), *sorted(kernel_dir.glob('*.cpp'))]
        subprocess.run([CXX, '-std=c++17', '-O2', '-DJOPT_SIM', '-D_FILE_OFFSET_BITS=64', *cls.DEFINES,
                        f'-I{kernel_dir}', f'-I{ROOT / "src/common"}',
                        *map(str, sources), '-o', str(cls.exe)], check=True)
        cls.header = register_header(cls.KERNEL, cls.PORTS)
        (cls.dir / 'hw.h').write_text(cls.header)

        # Counts whose z-score is N(0, 1), the dropped bands 65535 (as test_justounetsimple).
        kept, mean, inv_std = cls.model.read_preprocessing()
        z = np.random.default_rng(7).normal(0, 1, (cls.H, cls.W, 110)).astype(np.float32)
        raw = np.full((cls.H, cls.W, 120), 65535.0, dtype=np.float32)
        raw[:, :, kept] = z / inv_std + mean
        cls.cube = np.clip(np.rint(raw), 0, 65535).astype(np.uint16)
        np.save(cls.dir / 'cube.npy', cls.cube)
        cls.cube.tofile(cls.dir / 'cube.bip')
        cls.reference = cls.model.with_class(cls.model.classify_fixed(cls.cube)).astype(np.int32)
        np.save(cls.dir / 'reference_fixed.npy', cls.reference)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))
        cls.tmp.cleanup()

    def driver(self, *args, header='hw.h'):
        return subprocess.run([str(self.exe), '--header', str(self.dir / header), *map(str, args)],
                              capture_output=True, text=True)

    def test_whole_image_in_one_start_matches_the_integer_model(self):
        out = self.driver('--input', self.dir / 'cube.npy', '--reference', self.dir / 'reference_fixed.npy',
                          '--scores', self.dir / 'scores.npy', '--labels', self.dir / 'labels.npy')
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn('; 1 start', out.stdout)
        self.assertIn('PASS', out.stdout)
        np.testing.assert_array_equal(np.load(self.dir / 'scores.npy'), self.reference)
        labels = np.load(self.dir / 'labels.npy')
        self.assertEqual(labels.dtype, np.uint8)
        np.testing.assert_array_equal(labels, self.reference[..., 3])

    def test_strips_when_the_cube_does_not_fit_give_the_same_answer(self):
        # Room for two 32-row strips of 45 pixels, not for the whole cube.
        out = self.driver('--input', self.dir / 'cube.npy', '--reference', self.dir / 'reference_fixed.npy',
                          '--sim-memory', 800000)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn('3 starts of 32 rows', out.stdout)

    def test_raw_bip_with_hypso_capture_config_and_hypso_label_order(self):
        (self.dir / 'capture_config.ini').write_text(
            f'frame_count = {self.H}\nexposure = 47.32\nrow_count = {self.W}\naoi_y = 62\n')
        out = self.driver('--input', self.dir / 'cube.bip', '--capture-config', self.dir / 'capture_config.ini',
                          '--labels', self.dir / 'labels.bin', '--hypso-labels', '--strip-rows', 64)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        labels = np.fromfile(self.dir / 'labels.bin', dtype=np.uint8).reshape(self.H, self.W)
        np.testing.assert_array_equal(labels, 2 - self.reference[..., 3])   # 0 sea, 1 land, 2 cloud

    def test_a_wrong_answer_fails(self):
        wrong = self.reference.copy()
        wrong[40, 30, 1] += 1
        np.save(self.dir / 'wrong.npy', wrong)
        out = self.driver('--input', self.dir / 'cube.npy', '--reference', self.dir / 'wrong.npy')
        self.assertEqual(out.returncode, 2, out.stdout + out.stderr)
        self.assertIn('FAIL, 1 of', out.stdout)
        self.assertIn('row 40, column 30', out.stdout)

    def test_the_read_ports_come_from_the_header(self):
        out = self.driver('--input', self.dir / 'cube.npy')
        self.assertIn(f'{self.PORTS} read ports', out.stdout)
        # One port fewer in the header: the driver points the ones it has at the cube.
        last = f'DIN{self.PORTS - 1}_'
        (self.dir / 'fewer.h').write_text(''.join(l + '\n' for l in self.header.splitlines() if last not in l))
        out = self.driver('--input', self.dir / 'cube.npy', '--reference', self.dir / 'reference_fixed.npy',
                          header='fewer.h')
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(f'{self.PORTS - 1} read ports', out.stdout)

    def test_refuses_a_kernel_argument_it_does_not_set(self):
        (self.dir / 'extra.h').write_text(self.header + f'#define X{self.KERNEL.upper()}_CONTROL_ADDR_GAIN_DATA 0x5c\n')
        out = self.driver('--input', self.dir / 'cube.npy', header='extra.h')
        self.assertEqual(out.returncode, 1)
        self.assertIn('GAIN_DATA', out.stderr)

    def test_strips_must_be_whole_patches(self):
        out = self.driver('--input', self.dir / 'cube.npy', '--strip-rows', 40)
        self.assertEqual(out.returncode, 1)
        self.assertIn('multiple of 32', out.stderr)


@unittest.skipUnless(CXX and np is not None, 'needs a C++ compiler and numpy')
class ZCU104DriverTests(DriverTests, unittest.TestCase):
    KERNEL, PORTS = 'justounetsimple_opt', 4


@unittest.skipUnless(CXX and np is not None, 'needs a C++ compiler and numpy')
@unittest.skipUnless((ROOT / 'src/hls/justounetsimple_opt_zynq7030').is_dir(), 'no justounetsimple_opt_zynq7030 kernel')
class Zynq7030DriverTests(DriverTests, unittest.TestCase):
    KERNEL, PORTS, DEFINES = 'justounetsimple_opt_zynq7030', 2, ['-DJOPT_SIM_ZYNQ7030']


if __name__ == '__main__':
    unittest.main()
