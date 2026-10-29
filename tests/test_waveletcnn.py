"""waveletcnn: the numpy models against the C++ kernels and the OPU program. The board
notebook, shared by every model, is tests/test_notebooks.py's.
"""
import contextlib
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which('g++') or shutil.which('clang++')
CC = shutil.which('gcc') or shutil.which('clang')
OPT_SOURCE = ROOT / 'src/hls/waveletcnn_opt'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, 'needs numpy')
class WaveletCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'tools'))
        cls.model = load_module('waveletcnn_model', ROOT / 'tools/waveletcnn_model.py')
        cls.patching = load_module('patching', ROOT / 'tools/patching.py')
        cls.layers = cls.model.load_weights()

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / 'tools'))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()        # a notebook's namespace (a cycle) may still map cube.npy; Windows locks it
        self.tmp.cleanup()

    def raw_cube(self, h, w, seed=0, counts=False):
        """Raw values whose preprocessing gives N(0, 1); the dropped bands hold 1e6, so a
        wrong band selection blows the scores up. counts: whole numbers 0 .. 65535 like a
        capture's, the dropped bands 65535."""
        kept, mean, inv_std = self.model.read_preprocessing()
        z = np.random.default_rng(seed).normal(0, 1, (h, w, 110)).astype(np.float32)
        raw = np.full((h, w, 120), 65535.0 if counts else 1.0e6, dtype=np.float32)
        raw[:, :, kept] = z / inv_std + mean
        return np.clip(np.rint(raw), 0, 65535) if counts else raw

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_numpy_model_matches_the_cpp_kernel(self):
        source = self.dir / 'run_kernel.cpp'
        source.write_text("""
#include "waveletcnn.hpp"
#include <cstdio>
static float din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS];
static float dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES];
int main(int, char** argv) {
    FILE* f = fopen(argv[1], "rb"); if (fread(din, sizeof din, 1, f) != 1) return 1; fclose(f);
    waveletcnn(din, dout);
    f = fopen(argv[2], "wb"); fwrite(dout, sizeof dout, 1, f); fclose(f);
}
""")
        exe = self.dir / 'run_kernel.exe'
        subprocess.run([CXX, '-std=c++14', '-O2', f'-I{ROOT / "src/hls/waveletcnn"}',
                        str(source), str(ROOT / 'src/hls/waveletcnn/waveletcnn.cpp'),
                        '-o', str(exe)], check=True)
        x = np.random.default_rng(3).uniform(-3, 3, (32, 32, 110)).astype(np.float32)
        x.tofile(self.dir / 'in.bin')
        subprocess.run([str(exe), str(self.dir / 'in.bin'), str(self.dir / 'out.bin')], check=True)
        cpp = np.fromfile(self.dir / 'out.bin', dtype=np.float32).reshape(32, 32, 3)
        want = self.model.forward_patch(x, self.layers)
        np.testing.assert_allclose(cpp, want, rtol=1e-3, atol=1e-3)
        self.assertGreater(float(want.max() - want.min()), 1.0, 'the reference should not be constant')

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_fixed_point_model_matches_the_optimized_cpp_kernel(self):
        """classify_fixed predicts the board's answers: it must be the optimized kernels'
        arithmetic bit for bit (the ZCU104's and the Zynq-7000s'), the z-score and the edge
        patches included."""
        h, w = 37, 45                                     # 2 x 2 patches, the edge ones partial
        cube = self.raw_cube(h, w, seed=3, counts=True).astype(np.uint16)
        cube.tofile(self.dir / 'in.bin')
        want = self.model.classify_fixed(cube)
        kernels = {'waveletcnn_opt': 'uint32_t cycles; waveletcnn_opt(d, d, d, d, dout.data(), h, w, cycles)',
                   'waveletcnn_opt_zynq7020': 'waveletcnn_opt_zynq7020(d, d, dout.data(), h, w)',
                   'waveletcnn_opt_zynq7030': 'waveletcnn_opt_zynq7030(d, d, d, d, dout.data(), h, w)'}
        for name, call in kernels.items():
            with self.subTest(kernel=name):
                source_dir = ROOT / 'src/hls' / name
                source = self.dir / f'run_{name}.cpp'
                source.write_text(f"""
#include "{name}.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int, char** argv) {{
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    std::vector<cube_word_t> din(h * w * CUBE_BANDS / BANDS_PER_WORD);
    std::vector<score_word_t> dout(h * w);
    FILE* f = fopen(argv[1], "rb");
    if (fread(din.data(), sizeof(cube_word_t), din.size(), f) != din.size()) return 1;
    fclose(f);
    cube_word_t *d = din.data();   // hls::burst_maxi takes a plain pointer
    {call};
    f = fopen(argv[2], "wb"); fwrite(dout.data(), sizeof(score_word_t), dout.size(), f); fclose(f);
}}
""")
                exe = self.dir / f'run_{name}.exe'
                subprocess.run([CXX, '-std=c++14', '-O2', f'-I{source_dir}', f'-I{ROOT / "src/common"}',
                                str(source), str(source_dir / f'{name}.cpp'), '-o', str(exe)], check=True)
                out = self.dir / f'out_{name}.bin'
                subprocess.run([str(exe), str(self.dir / 'in.bin'), str(out), str(h), str(w)], check=True)
                cpp = np.fromfile(out, dtype=np.int32).reshape(h, w, 4)
                np.testing.assert_array_equal(cpp, self.model.with_class(want))
        # The float model on the same int8 input: only the rounding between the layers differs.
        scores = want * 2.0 ** -self.model.OUT_FRAC
        floats = np.zeros((h, w, 3), dtype=np.float32)
        for patch, tile in self.patching.iter_patches(cube, pad_mode='edge'):
            x = self.model.preprocess_fixed(patch) * np.float32(2.0 ** -self.model.IN_FRAC)
            self.patching.place(floats, self.model.forward_patch(x, self.layers), tile)
        np.testing.assert_allclose(scores, floats, atol=0.02)
        self.assertGreater(float(scores.max() - scores.min()), 1.0, 'the reference should not be constant')

    @unittest.skipUnless(CXX and CC, 'no C and C++ compiler')
    def test_opu_classify_runs_the_kernel_and_checks_it(self):
        """software/opu/.../classify.c, the board's program for the ZedBoard, against a fake
        board that runs the 7020 kernel in software: the cube in, the scores out, bit-exact,
        in one start and in strips (a reserved memory too small for the cube), checked
        against the labels."""
        name = 'waveletcnn_opt_zynq7020'
        prefix = '#define XWAVELETCNN_OPT_ZYNQ7020_CONTROL_'
        header = ''.join(f'{prefix}{key} {value}\n' for key, value in (
            ('ADDR_AP_CTRL', '0x00'), ('ADDR_DIN0_DATA', '0x10'), ('BITS_DIN0_DATA', '64'),
            ('ADDR_DIN1_DATA', '0x1c'), ('BITS_DIN1_DATA', '64'), ('ADDR_DOUT_DATA', '0x28'),
            ('BITS_DOUT_DATA', '64'), ('ADDR_HEIGHT_DATA', '0x34'), ('ADDR_WIDTH_DATA', '0x3c')))
        flow = load_module('flow', ROOT / 'main.py')
        (self.dir / 'kernel_regs.h').write_text(flow.opu_registers(header))
        (self.dir / 'fake_board.cpp').write_text(r"""
#include "waveletcnn_opt_zynq7020.hpp"
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
    waveletcnn_opt_zynq7020(d, d, out, h, w);
    regs[REG_AP_CTRL / 4] = 4 | 2;              // done, idle
    fprintf(stderr, "fake board: a start, %d x %d\n", h, w);
}
uint32_t reg_read(struct board *, unsigned offset) {
    const uint32_t v = regs[offset / 4];
    if (offset == REG_AP_CTRL) regs[offset / 4] &= ~2u;   // done clears on read
    return v;
}
}
""")
        source_dir = ROOT / 'src/hls' / name
        program = ROOT / 'software/opu/justounetsimple_opt/classify.c'
        exe = self.dir / 'classify.exe'
        subprocess.run([CC, '-std=c99', '-O2', '-Wall', '-Werror', '-DFAKE_BOARD', f'-I{self.dir}', '-c',
                        str(program), '-o', str(self.dir / 'classify.o')], check=True)
        subprocess.run([CXX, '-std=c++14', '-O2', f'-I{source_dir}', f'-I{ROOT / "src/common"}', f'-I{self.dir}',
                        str(self.dir / 'fake_board.cpp'), str(source_dir / f'{name}.cpp'),
                        str(self.dir / 'classify.o'), '-o', str(exe)], check=True)

        h, w = 70, 45                                     # 3 x 2 patches, the edge ones partial
        cube = self.raw_cube(h, w, seed=5, counts=True).astype(np.uint16)
        want = self.model.with_class(self.model.classify_fixed(cube))
        np.save(self.dir / 'cube.npy', cube)
        # labels.npy next to the cube, which classify needs: the accuracy and each class's IoU
        # over the labelled pixels (3: unlabeled), as the notebooks compute them.
        rng = np.random.default_rng(6)
        classes = want[..., 3]
        labels = np.where(rng.random((h, w)) < 0.8, classes, rng.integers(0, 4, (h, w)))
        known = labels < 3
        accuracy = f'accuracy against {self.dir / "labels.npy"}: {100 * (classes[known] == labels[known]).mean():.2f}% ' \
                   f'of {known.sum()} labelled pixels'
        iou = ', '.join(f'{name} {((classes == c) & (labels == c))[known].sum() / ((classes == c) | (labels == c))[known].sum():.3f}'
                        for c, name in enumerate(('cloud', 'land', 'sea')))
        self.assertLess(known.sum(), h * w, 'some unlabeled pixels, to be skipped')
        np.save(self.dir / 'labels.npy', labels)

        def classify(cube_file, memory=64 << 20):
            return subprocess.run([str(exe), str(self.dir / cube_file), str(self.dir / 'scores.npy')],
                                  capture_output=True, text=True, env={**os.environ, 'FAKE_MEMORY': str(memory)})

        # All of it at once; then 32 rows at a time: room for 64 rows of the cube, but only for
        # 60 with their scores (a strip size that forgets the scores overruns the memory).
        for memory, starts in ((64 << 20, 1), (4096 + 64 * w * 120 * 2, 3)):
            with self.subTest(memory=memory):
                result = classify('cube.npy', memory)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f'in {starts} start', result.stdout)
                self.assertEqual(result.stderr.count('fake board: a start'), starts)
                np.testing.assert_array_equal(np.load(self.dir / 'scores.npy'), want)
                self.assertIn(accuracy, result.stdout)
                self.assertIn(f'IoU {iou}', result.stdout)

    def test_with_class_picks_the_first_highest_score(self):
        scores = np.array([[[5, 7, 7], [9, 1, 9], [-3, -2, -4]]], dtype=np.int32)
        np.testing.assert_array_equal(self.model.with_class(scores)[..., 3], [[1, 0, 1]])
        np.testing.assert_array_equal(self.model.with_class(scores)[..., :3], scores)

    def test_kernel_input_is_int8_and_never_minus_128(self):
        x = self.model.quantize_input(np.array([-100.0, -3.99, 0.0, 3.99, 100.0]))
        self.assertEqual(x.dtype, np.int8)
        np.testing.assert_array_equal(x, [-127, -127, 0, 127, 127])
        raw = np.zeros((1, 1, 120), dtype=np.uint16)
        raw[..., 8:118] = 65535
        self.assertEqual(int(self.model.preprocess_fixed(raw).max()), 127)

    def test_integer_z_score_is_within_one_unit_of_the_float_one(self):
        cube = self.raw_cube(16, 16, counts=True)
        kept, mean, inv_std = self.model.read_preprocessing()
        floats = self.model.quantize_input(self.model.preprocess(cube, kept, mean, inv_std))
        ints = self.model.preprocess_fixed(cube.astype(np.uint16))
        self.assertLessEqual(int(np.abs(ints.astype(int) - floats).max()), 1)

    def test_model_uses_the_kernel_formats(self):
        """The numpy model and tools/models.py hard-code what waveletcnn_opt.hpp defines."""
        header = (OPT_SOURCE / 'waveletcnn_opt.hpp').read_text(encoding='utf-8')

        def define(name):
            return int(re.search(rf'constexpr int {name} = (\d+);', header).group(1))

        models = load_module('models', ROOT / 'tools/models.py')
        self.assertEqual(models.MODELS['waveletcnn']['out_frac'], define('SCORE_FRAC_BITS'))
        self.assertEqual(define('CUBE_BANDS'), 120)
        self.assertIn('hls::vector<int32_t, 4> score_word_t', header)
        for model_name, kernel_name in (('IN_FRAC', 'ZSCORE_FRAC_BITS'), ('ACT_FRAC', 'ACTIVATION_FRAC_BITS'),
                                        ('OUT_FRAC', 'SCORE_FRAC_BITS')):
            self.assertEqual(getattr(self.model, model_name), define(kernel_name), kernel_name)