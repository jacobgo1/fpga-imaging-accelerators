"""software/opu/classify.c, the program every optimized kernel runs on the OPU system's
PetaLinux, against a fake board that runs the kernel's C++ in software: the cube in, the
scores out, bit-exact, in one start and in strips (a reserved memory too small for the
cube), checked against the labels.
"""
import gc
import os
from pathlib import Path
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


@unittest.skipIf(np is None, 'needs numpy')
@unittest.skipUnless(CXX and CC, 'no C and C++ compiler')
class OpuClassifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.flow = load_module('flow', ROOT / 'main.py')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()
        self.tmp.cleanup()

    def build(self, kernel):
        """classify on the fake board, with KERNEL's registers."""
        prefix = f'#define X{kernel.upper()}_CONTROL_'
        header = ''.join(f'{prefix}{key} {value}\n' for key, value in (
            ('ADDR_AP_CTRL', '0x00'), ('ADDR_DIN0_DATA', '0x10'), ('BITS_DIN0_DATA', '64'),
            ('ADDR_DIN1_DATA', '0x1c'), ('BITS_DIN1_DATA', '64'), ('ADDR_DOUT_DATA', '0x28'),
            ('BITS_DOUT_DATA', '64'), ('ADDR_HEIGHT_DATA', '0x34'), ('ADDR_WIDTH_DATA', '0x3c')))
        (self.dir / 'kernel_regs.h').write_text(self.flow.opu_registers(header))
        subprocess.run([CC, '-std=c99', '-O2', '-Wall', '-Werror', '-DFAKE_BOARD', f'-I{self.dir}', '-c',
                        str(ROOT / 'software/opu/classify.c'), '-o', str(self.dir / 'classify.o')], check=True)
        source_dir = kernel_dir(kernel)
        return compile_cpp(self.dir, f'classify_{kernel}', FAKE_BOARD.replace('KERNEL', kernel),
                           [source_dir, self.dir], [source_dir / f'{kernel}.cpp', self.dir / 'classify.o'])

    def cube_and_labels(self, kernel):
        """cube.npy, 3 x 2 patches with the edge ones partial, and labels.npy, mostly the
        kernel's classes (3: unlabeled). Returns them, the scores classify must write, and
        the accuracy and IoU it must print (as the notebooks compute them)."""
        model = load_module(KERNELS[kernel], ROOT / 'tools' / f'{KERNELS[kernel]}.py')
        h, w = 70, 45
        cube = raw_cube(model, h, w, seed=5, counts=True).astype(np.uint16)
        want = model.with_class(model.classify_fixed(cube))
        rng = np.random.default_rng(6)
        classes = want[..., 3]
        labels = np.where(rng.random((h, w)) < 0.8, classes, rng.integers(0, 4, (h, w)))
        known = labels < 3
        self.assertLess(known.sum(), h * w, 'some unlabeled pixels, to be skipped')
        accuracy = f'accuracy against {self.dir / "labels.npy"}: {100 * (classes[known] == labels[known]).mean():.2f}% ' \
                   f'of {known.sum()} labelled pixels'
        iou = ', '.join(f'{name} {((classes == c) & (labels == c))[known].sum() / ((classes == c) | (labels == c))[known].sum():.3f}'
                        for c, name in enumerate(('cloud', 'land', 'sea')))
        np.save(self.dir / 'cube.npy', cube)
        np.save(self.dir / 'labels.npy', labels)
        return cube, labels, want, accuracy, iou

    def classify(self, exe, cube_file, memory=64 << 20):
        return subprocess.run([str(exe), str(self.dir / cube_file), str(self.dir / 'scores.npy')],
                              capture_output=True, text=True, env={**os.environ, 'FAKE_MEMORY': str(memory)})

    def test_every_kernel_bit_exact_in_one_start_and_in_strips(self):
        for kernel in KERNELS:
            exe = self.build(kernel)
            cube, _, want, accuracy, iou = self.cube_and_labels(kernel)
            # All of it at once; then 32 rows at a time: room for 64 rows of the cube, but only
            # for 60 with their scores (a strip size that forgets the scores overruns the memory).
            for memory, starts in ((64 << 20, 1), (4096 + 64 * cube.shape[1] * 120 * 2, 3)):
                with self.subTest(kernel=kernel, memory=memory):
                    result = self.classify(exe, 'cube.npy', memory)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f'in {starts} start', result.stdout)
                    self.assertEqual(result.stderr.count('fake board: a start'), starts)
                    np.testing.assert_array_equal(np.load(self.dir / 'scores.npy'), want)
                    self.assertIn(accuracy, result.stdout)
                    self.assertIn(f'IoU {iou}', result.stdout)

    def test_says_what_is_wrong_with_the_labels_or_the_cube(self):
        kernel = 'justounetsimple_opt_zynq7020'
        exe = self.build(kernel)
        cube, labels, _, accuracy, _ = self.cube_and_labels(kernel)
        # uint8 labels, as a capture's own are (prepare writes int64)
        np.save(self.dir / 'labels.npy', labels.astype(np.uint8))
        result = self.classify(exe, 'cube.npy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(accuracy, result.stdout)
        # Labels of another size, or none: said, exit 1.
        np.save(self.dir / 'labels.npy', labels[:-1])
        result = self.classify(exe, 'cube.npy')
        self.assertEqual(result.returncode, 1)
        self.assertIn('the labels are not H x W for this cube', result.stderr)
        (self.dir / 'labels.npy').unlink()
        result = self.classify(exe, 'cube.npy')
        self.assertEqual(result.returncode, 1)
        self.assertIn(f'no {self.dir / "labels.npy"}', result.stderr)
        # Not the cube's type: said, exit 1.
        np.save(self.dir / 'floats.npy', cube.astype(np.float32))
        result = self.classify(exe, 'floats.npy', 1048576)
        self.assertEqual(result.returncode, 1)
        self.assertIn("expected '<u2'", result.stderr)


if __name__ == '__main__':
    unittest.main()
