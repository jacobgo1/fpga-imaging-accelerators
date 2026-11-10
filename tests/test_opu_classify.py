"""software/opu/classify.c, the program every optimized kernel runs on the OPU system's
PetaLinux, against a fake board that runs the kernel's C++ in software: the raw .bip cube in,
the scores out, bit-exact, in one start and in strips (a reserved memory too small for the
cube). And software/opu/benchmark.sh, whose times and scores tools/opu_results.py turns into
the report on the PC, with accuracy and IoU against the labels; and tools/image.py's cube.bip.
"""
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from helpers import CC, CXX, ROOT, compile_cpp, kernel_dir, load_module, raw_cube

try:
    import numpy as np
except ImportError:
    np = None

# Two kernels of different models, both the 7020's (two read ports), and their models.
KERNELS = {'justounetsimple_opt_zynq7020': 'justounetsimple_model', 'waveletcnn_opt_zynq7020': 'waveletcnn_model'}

# classify.c's board calls, in software: a start runs KERNEL on the fake reserved memory.
FAKE_BOARD = r"""
#include "KERNEL.hpp"
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "kernel_regs.h"
struct board { volatile uint32_t *regs; uint8_t *mem; uint64_t phys, size; };
static uint32_t regs[64];
static uint8_t *memory;
static uint64_t size;
static const uint64_t PHYS = 0x10000000;   // where the fake reserved memory is for the kernel
static void *at(unsigned offset, size_t bytes) {   // a kernel address -> the fake memory
    const uint64_t a = regs[offset / 4] | (uint64_t)regs[offset / 4 + 1] << 32;
    if (a < PHYS || a + bytes > PHYS + size) { fprintf(stderr, "fake board: address %llx outside\n", (unsigned long long)a); exit(3); }
    return memory + (a - PHYS);
}
void open_board(struct board *b) {
    size = strtoull(getenv("FAKE_MEMORY"), 0, 0);
    memory = (uint8_t *)calloc(size, 1);
    b->regs = regs; b->mem = memory; b->phys = PHYS; b->size = size;
    regs[REG_AP_CTRL / 4] = 4;                  // idle
}
void reg_write(struct board *, unsigned offset, uint32_t value) {
    regs[offset / 4] = value;
    if (offset != REG_AP_CTRL || !(value & 1)) return;
    const int dins[] = REG_DINS;                // every port must read the same cube
    const int h = regs[REG_HEIGHT / 4], w = regs[REG_WIDTH / 4];
    if (regs[dins[0] / 4] != regs[dins[1] / 4]) exit(4);
    cube_word_t *d = (cube_word_t *)at(dins[0], (size_t)h * w * CUBE_BANDS * 2);
    score_word_t *out = (score_word_t *)at(REG_DOUT, (size_t)h * w * sizeof(score_word_t));
    KERNEL(d, d, out, h, w);
#ifdef REG_CYCLES
    regs[REG_CYCLES / 4] = 1000 + h * w;        // a count the tests can predict
#endif
    regs[REG_AP_CTRL / 4] = 4 | 2;              // done, idle
    fprintf(stderr, "fake board: a start, %d x %d\n", h, w);
}
uint32_t reg_read(struct board *, unsigned offset) {
    const uint32_t v = regs[offset / 4];
    if (offset == REG_AP_CTRL) regs[offset / 4] &= ~2u;   // done clears on read
    return v;
}
}
"""


def expected_counts(classes, labels):
    """What the PC's report must find: labelled, correct, and per class both / either."""
    known = (labels >= 0) & (labels <= 2)
    c, l = classes[known], labels[known]
    return known.sum(), (c == l).sum(), [((c == k) & (l == k)).sum() for k in range(3)], \
        [((c == k) | (l == k)).sum() for k in range(3)]


@unittest.skipIf(np is None, 'needs numpy')
@unittest.skipUnless(CXX and CC, 'no C and C++ compiler')
class OpuClassifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.flow = load_module('flow', ROOT / 'main.py')
        cls.results = load_module('opu_results', ROOT / 'tools/opu_results.py')
        cls.image = load_module('image', ROOT / 'tools/image.py')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()
        self.tmp.cleanup()

    def build(self, kernel, cycles=False):
        """classify on the fake board, with KERNEL's registers (and a cycle counter if cycles)."""
        prefix = f'#define X{kernel.upper()}_CONTROL_'
        header = ''.join(f'{prefix}{key} {value}\n' for key, value in (
            ('ADDR_AP_CTRL', '0x00'), ('ADDR_DIN0_DATA', '0x10'), ('BITS_DIN0_DATA', '64'),
            ('ADDR_DIN1_DATA', '0x1c'), ('BITS_DIN1_DATA', '64'), ('ADDR_DOUT_DATA', '0x28'),
            ('BITS_DOUT_DATA', '64'), ('ADDR_HEIGHT_DATA', '0x34'), ('ADDR_WIDTH_DATA', '0x3c'))
            + ((('ADDR_CYCLES_DATA', '0x44'),) if cycles else ()))
        (self.dir / 'kernel_regs.h').write_text(self.flow.opu_registers(header))
        subprocess.run([CC, '-std=c99', '-O2', '-Wall', '-Werror', '-DFAKE_BOARD', f'-I{self.dir}', '-c',
                        str(ROOT / 'software/opu/classify.c'), '-o', str(self.dir / 'classify.o')], check=True)
        source_dir = kernel_dir(kernel)
        return compile_cpp(self.dir, f'classify_{kernel}_{int(cycles)}', FAKE_BOARD.replace('KERNEL', kernel),
                           [source_dir, self.dir], [source_dir / f'{kernel}.cpp', self.dir / 'classify.o'])

    def cube(self, kernel, folder=None):
        """An image folder as tools/image.py makes it: cube.npy, labels.npy (mostly the kernel's
        classes, 3: unlabeled), cube.bip and size.txt; 3 x 2 patches with the edge ones partial.
        Returns the cube, the labels and the scores the board must give."""
        folder = folder or self.dir
        folder.mkdir(parents=True, exist_ok=True)
        model = load_module(KERNELS[kernel], ROOT / 'tools' / f'{KERNELS[kernel]}.py')
        h, w = 70, 45
        cube = raw_cube(model, h, w, seed=5, counts=True).astype(np.uint16)
        want = model.with_class(model.classify_fixed(cube))
        rng = np.random.default_rng(6)
        labels = np.where(rng.random((h, w)) < 0.8, want[..., 3], rng.integers(0, 4, (h, w)))
        np.save(folder / 'cube.npy', cube)
        np.save(folder / 'labels.npy', labels)
        self.image.write_bip(folder, cube)
        return cube, labels, want

    def classify(self, exe, *args, memory=64 << 20):
        return subprocess.run([str(exe), str(self.dir / 'cube.bip'), str(self.dir / 'scores.npy'), *map(str, args)],
                              capture_output=True, text=True, env={**os.environ, 'FAKE_MEMORY': str(memory)})

    def test_image_bip_is_the_npy_data_and_its_size(self):
        cube, *_ = self.cube(next(iter(KERNELS)))
        self.assertEqual((self.dir / 'cube.bip').read_bytes(), cube.tobytes())
        self.assertEqual((self.dir / 'size.txt').read_bytes(), b'70 45\n')
        # bip on a folder made before: the same files from its cube.npy.
        (self.dir / 'cube.bip').unlink()
        (self.dir / 'size.txt').unlink()
        saved, sys.argv = sys.argv, ['image.py', 'bip', str(self.dir)]
        try:
            self.image.main()
        finally:
            sys.argv = saved
        gc.collect()
        self.assertEqual((self.dir / 'cube.bip').read_bytes(), cube.tobytes())
        self.assertEqual((self.dir / 'size.txt').read_text(), '70 45\n')

    def test_every_kernel_bit_exact_in_one_start_and_in_strips(self):
        for kernel in KERNELS:
            exe = self.build(kernel)
            cube, _, want = self.cube(kernel)
            h, w = cube.shape[:2]
            # All of it at once; then 32 rows at a time: room for 64 rows of the cube, but only
            # for 60 with their scores (a strip size that forgets the scores overruns the memory).
            for memory, starts in ((64 << 20, 1), (4096 + 64 * w * 120 * 2, 3)):
                with self.subTest(kernel=kernel, memory=memory):
                    result = self.classify(exe, h, w, memory=memory)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f'in {starts} start', result.stdout)
                    self.assertEqual(result.stderr.count('fake board: a start'), starts)
                    np.testing.assert_array_equal(np.load(self.dir / 'scores.npy'), want)

    def test_says_what_is_wrong_with_the_arguments(self):
        exe = self.build('justounetsimple_opt_zynq7020')
        cube, *_ = self.cube('justounetsimple_opt_zynq7020')
        h, w = cube.shape[:2]
        # Sizes that do not match the file: said, and the kernel never starts.
        result = self.classify(exe, h, w + 1)
        self.assertEqual(result.returncode, 1)
        self.assertIn(f'but {h} x {w + 1} x 120 uint16 is', result.stderr)
        self.assertNotIn('fake board: a start', result.stderr)
        result = self.classify(exe, h)
        self.assertEqual(result.returncode, 2)
        self.assertIn('CUBE.bip SCORES.npy HEIGHT WIDTH', result.stderr)
        result = self.classify(exe, h, 'x')
        self.assertEqual(result.returncode, 2)
        self.assertIn('WIDTH must be a positive whole number', result.stderr)

    def test_own_cycle_count_added_up_over_the_strips(self):
        kernel = 'justounetsimple_opt_zynq7020'
        cube, *_ = self.cube(kernel)
        h, w = cube.shape[:2]
        # 70 rows in strips of 32: 32, 32 and 6.
        result = self.classify(self.build(kernel, cycles=True), h, w, memory=4096 + 64 * w * 120 * 2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f'own count:          {3000 + h * w} cycles', result.stdout)
        # A kernel without the counter says nothing about it.
        result = self.classify(self.build(kernel), h, w)
        self.assertNotIn('own count', result.stdout)

    def test_one_run_checked_on_the_pc(self):
        kernel = 'waveletcnn_opt_zynq7020'
        _, labels, want = self.cube(kernel)
        np.save(self.dir / 'scores.npy', want)
        labelled, correct, both, either = expected_counts(want[..., 3], labels)
        self.assertLess(labelled, labels.size, 'some unlabeled pixels, to be skipped')
        saved, sys.argv = sys.argv, ['opu_results.py', '--scores', str(self.dir / 'scores.npy'),
                                     '--labels', str(self.dir / 'labels.npy')]
        try:
            from io import StringIO
            from contextlib import redirect_stdout
            with redirect_stdout(StringIO()) as out:
                self.results.main()
        finally:
            sys.argv = saved
        self.assertIn(f'accuracy: {100 * correct / labelled:.2f}% of {labelled} labelled pixels', out.getvalue())
        self.assertIn(f'cloud {both[0] / either[0]:.3f}, land {both[1] / either[1]:.3f}', out.getvalue())

    @unittest.skipUnless(shutil.which('sh') and shutil.which('awk'), 'no sh and awk')
    def test_benchmark_runs_every_kernel_on_every_image_and_the_pc_reports(self):
        """software/opu/benchmark.sh on a fake board: each kernel's classify (one with its own
        count, one without), a kernel whose load.sh fails, a kernel with no folder, two images;
        then tools/opu_results.py's report from what it wrote."""
        top = self.dir / 'board'
        firmware = top / 'firmware'
        firmware.mkdir(parents=True)
        shutil.copy2(ROOT / 'software/opu/benchmark.sh', top)
        for kernel, cycles in zip(KERNELS, (True, False)):
            folder = top / f'{kernel}_opu'
            folder.mkdir()
            shutil.copy2(self.build(kernel, cycles), folder / 'classify')
            (folder / 'load.sh').write_text('echo "FCLK_CLK0: 100000 kHz, built for 100000 kHz"\n')
        broken = top / 'justoliunet_opt_zynq7020_opu'
        broken.mkdir()
        (broken / 'classify').write_text('')
        (broken / 'load.sh').write_text('echo "md5sum: FAILED"; exit 1\n')
        # The same cube and labels for both images (the labels follow the first kernel's classes).
        cube, labels, _ = self.cube(next(iter(KERNELS)), top / 'one')
        shutil.copytree(top / 'one', top / 'two')
        h, w = labels.shape
        env = {**os.environ, 'FIRMWARE': str(firmware), 'FAKE_MEMORY': str(64 << 20), 'IMAGES': 'one two',
               'KERNELS': ' '.join(list(KERNELS) + ['justoliunet_opt_zynq7020', 'sp_unet_small_opt_zynq7020'])}
        result = subprocess.run(['sh', str(top / 'benchmark.sh')], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('skipped (no KERNEL_opu/classify): sp_unet_small_opt_zynq7020', result.stdout)
        self.assertTrue((firmware / 'load.sh').exists(), 'the kernel files went to the firmware folder')
        out = next((top / 'benchmark_results').iterdir())
        csv_rows = (out / 'times.csv').read_text().splitlines()
        self.assertEqual(len(csv_rows), 1 + 3 * 2)
        self.assertEqual(sum('md5sum: FAILED' in row for row in csv_rows), 2)
        for kernel in KERNELS:
            model = load_module(KERNELS[kernel], ROOT / 'tools' / f'{KERNELS[kernel]}.py')
            want = model.with_class(model.classify_fixed(cube))
            for image in ('one', 'two'):
                np.testing.assert_array_equal(np.load(out / 'scores' / kernel / f'{image}.npy'), want)

        saved, sys.argv = sys.argv, ['opu_results.py', str(out), '--images', str(top)]
        try:
            from io import StringIO
            from contextlib import redirect_stdout
            with redirect_stdout(StringIO()):
                self.results.main()
        finally:
            sys.argv = saved
        report = (out / 'report.md').read_text()
        rows = {line.split('|')[1].strip(): line for line in report.splitlines() if line.startswith('| `')}
        counted, uncounted = KERNELS
        self.assertIn(f'| `{counted}` | 100.0 | 2 | {(1000 + h * w) / 6:.2f} a patch | {(1000 + h * w) / 1e5:.3f} |',
                      rows[f'`{counted}`'])
        self.assertIn(f'| `{uncounted}` | 100.0 | 2 | - | - |', rows[f'`{uncounted}`'])
        for kernel in KERNELS:
            model = load_module(KERNELS[kernel], ROOT / 'tools' / f'{KERNELS[kernel]}.py')
            labelled, correct, both, either = expected_counts(model.with_class(model.classify_fixed(cube))[..., 3], labels)
            accuracy = 100 * correct / labelled
            iou = [b / e for b, e in zip(both, either)]
            self.assertIn(f'| {accuracy:.2f}% | {accuracy:.2f}% | {iou[0]:.3f} | {iou[1]:.3f} | {iou[2]:.3f} '
                          f'| {sum(iou) / 3:.3f} |', rows[f'`{kernel}`'])
        self.assertNotIn('`justoliunet_opt_zynq7020`', rows)
        self.assertEqual(report.count('failed: load.sh: md5sum: FAILED'), 2)
        self.assertIn(f'one ({h} x {w}), two ({h} x {w})', report)


if __name__ == '__main__':
    unittest.main()
