"""software/opu/: classify, the program every optimized kernel runs on the OPU system's
PetaLinux, against a fake board that runs the kernel's C++: its core (src/classify.cpp)
reads the cube, starts the kernel in one start or in strips, and writes HYPSO's labels
and the scores, bit-exact, checked against true labels. Also make_kernel_header.sh (the
kernel's HLS driver under shared names) and benchmark.sh.

The board's own parts (UIO, the HLS driver, spdlog, libconfig, argparse) are built by
scripts/opu.sh for the board and are not compiled here.
"""
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from helpers import CXX, ROOT, kernel_dir, load_module, raw_cube

try:
    import numpy as np
except ImportError:
    np = None

BASH = shutil.which('bash')
OPU = ROOT / 'software/opu'
# Two kernels of different models, both the 7020's (two read ports), and their models.
KERNELS = {'justounetsimple_opt_zynq7020': 'justounetsimple_model', 'waveletcnn_opt_zynq7020': 'waveletcnn_model'}

# classify's main() on a fake board: the reserved memory is two vectors, a start runs KERNEL.
# The same options as the real one; the maps' sizes from FAKE_CUBE_MAP and FAKE_SCORES_MAP,
# a raw cube's size from --height and --width (the real one's capture config).
FAKE_BOARD = r"""
#include "classify.h"
#include "KERNEL.hpp"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
int main(int argc, char **argv) {
    std::vector<uint8_t> cube_map(std::strtoull(std::getenv("FAKE_CUBE_MAP"), 0, 0));
    std::vector<uint8_t> scores_map(std::strtoull(std::getenv("FAKE_SCORES_MAP"), 0, 0));
    const uint64_t CUBE = 0x20000000, SCORES = 0x29600000;   // where they are for the kernel
    Outputs outputs;
    std::string input;
    long height = 0, width = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string option = argv[i], value = argv[i + 1];
        if (option == "--input") input = value;
        else if (option == "--output") outputs.labels = value;
        else if (option == "--labels") outputs.truth = value;
        else if (option == "--scores") outputs.scores = value;
        else if (option == "--height") height = std::atol(value.c_str());
        else if (option == "--width") width = std::atol(value.c_str());
    }
    auto start = [&](uint64_t cube, uint64_t scores, uint32_t h, uint32_t w) -> uint64_t {
        if (cube != CUBE || scores != SCORES) { std::fprintf(stderr, "fake board: wrong address\n"); std::exit(3); }
        if ((size_t)h * w * CUBE_BANDS * 2 > cube_map.size() || (size_t)h * w * sizeof(score_word_t) > scores_map.size()) {
            std::fprintf(stderr, "fake board: %u x %u overruns the memory\n", h, w); std::exit(4);
        }
        cube_word_t *d = (cube_word_t *)cube_map.data();
        KERNEL(d, d, (score_word_t *)scores_map.data(), h, w);
        std::fprintf(stderr, "fake board: a start, %u x %u\n", h, w);
        return 1000 + h * w;                            // a count the tests can predict
    };
    try {
        Cube cube = openCube(input, height, width);
        Result result = classify(cube, {cube_map.data(), CUBE, cube_map.size()},
                                 {scores_map.data(), SCORES, scores_map.size()}, start, outputs);
        result.has_cycles = std::getenv("FAKE_CYCLES") != nullptr;
        if (!result.has_cycles) result.cycles = 0;
        std::printf("kernel:             KERNEL, built for 100.000 MHz\n");
        std::fflush(stdout);
        printResult(result, outputs, std::cout);
        std::cout << "wall clock:            0.500 s\n";
    } catch (const std::exception &e) {
        std::fprintf(stderr, "classify: %s\n", e.what());
        return 1;
    }
}
"""


@unittest.skipIf(np is None, 'needs numpy')
@unittest.skipUnless(CXX, 'no C++ compiler')
class ClassifyTests(unittest.TestCase):
    built = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()
        self.tmp.cleanup()

    @classmethod
    def build(cls, kernel):
        """classify on the fake board running KERNEL, compiled once for all tests."""
        if kernel not in cls.built:
            out = Path(tempfile.mkdtemp())
            source = out / 'fake_board.cpp'
            source.write_text(FAKE_BOARD.replace('KERNEL', kernel))
            folder = kernel_dir(kernel)
            exe = out / 'classify'
            subprocess.run([CXX, '-std=c++17', '-O2', f'-I{OPU / "src"}', f'-I{folder}', f'-I{ROOT / "src/common"}',
                            str(source), str(OPU / 'src/classify.cpp'), str(folder / f'{kernel}.cpp'),
                            '-o', str(exe)], check=True)
            cls.built[kernel] = exe
        return cls.built[kernel]

    def cube_and_labels(self, kernel):
        """cube.npy, 3 x 2 patches with the edge ones partial, and labels.npy, mostly the
        kernel's classes (3: unlabeled). Returns them, the scores the kernel must write, and
        the accuracy and IoU classify must print (as the notebooks compute them)."""
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

    def classify(self, exe, *args, cube_map=64 << 20, scores_map=16 << 20, cycles=False):
        env = {**os.environ, 'FAKE_CUBE_MAP': str(cube_map), 'FAKE_SCORES_MAP': str(scores_map)}
        if cycles:
            env['FAKE_CYCLES'] = '1'
        return subprocess.run([str(exe), '--output', str(self.dir / 'labels.bin'), *[str(a) for a in args]],
                              capture_output=True, text=True, env=env)

    def test_every_kernel_bit_exact_in_one_start_and_in_strips(self):
        for kernel in KERNELS:
            exe = self.build(kernel)
            cube, _, want, accuracy, iou = self.cube_and_labels(kernel)
            w = cube.shape[1]
            # All of it at once; then 32 rows at a time: the cube map has room for 64 rows,
            # the scores map for 40 (a strip size that forgets the scores overruns it).
            for maps, starts in (((64 << 20, 16 << 20), 1), ((64 * w * 240, 40 * w * 16), 3)):
                with self.subTest(kernel=kernel, maps=maps):
                    result = self.classify(exe, '--input', self.dir / 'cube.npy', '--labels', self.dir / 'labels.npy',
                                           '--scores', self.dir / 'scores.npy', cube_map=maps[0], scores_map=maps[1])
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f'in {starts} start', result.stdout)
                    self.assertEqual(result.stderr.count('fake board: a start'), starts)
                    np.testing.assert_array_equal(np.load(self.dir / 'scores.npy'), want)
                    self.assertIn(accuracy, result.stdout)
                    self.assertIn(f'IoU {iou}', result.stdout)
                    # HYPSO's labels: sea 0, land 1, cloud 2 (the kernel's 2, 1, 0).
                    labels = np.fromfile(self.dir / 'labels.bin', dtype=np.uint8).reshape(cube.shape[:2])
                    np.testing.assert_array_equal(labels, 2 - want[..., 3])

    def test_a_raw_bip_cube_takes_its_size_from_the_capture(self):
        kernel = 'justounetsimple_opt_zynq7020'
        exe = self.build(kernel)
        cube, _, want, _, _ = self.cube_and_labels(kernel)
        cube.tofile(self.dir / 'binned_cube.bip')
        h, w = cube.shape[:2]
        result = self.classify(exe, '--input', self.dir / 'binned_cube.bip', '--height', h, '--width', w)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        labels = np.fromfile(self.dir / 'labels.bin', dtype=np.uint8).reshape(h, w)
        np.testing.assert_array_equal(labels, 2 - want[..., 3])
        self.assertNotIn('accuracy', result.stdout, 'no labels given, none checked')
        result = self.classify(exe, '--input', self.dir / 'binned_cube.bip', '--height', h + 1, '--width', w)
        self.assertEqual(result.returncode, 1)
        self.assertIn('but it only contains', result.stderr)

    def test_says_what_is_wrong_with_the_labels_or_the_cube(self):
        kernel = 'justounetsimple_opt_zynq7020'
        exe = self.build(kernel)
        cube, labels, _, accuracy, _ = self.cube_and_labels(kernel)
        # uint8 labels, as a capture's own are (prepare writes int64)
        np.save(self.dir / 'labels.npy', labels.astype(np.uint8))
        result = self.classify(exe, '--input', self.dir / 'cube.npy', '--labels', self.dir / 'labels.npy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(accuracy, result.stdout)
        np.save(self.dir / 'labels.npy', labels[:-1])
        result = self.classify(exe, '--input', self.dir / 'cube.npy', '--labels', self.dir / 'labels.npy')
        self.assertEqual(result.returncode, 1)
        self.assertIn('the labels are not H x W for this cube', result.stderr)
        np.save(self.dir / 'floats.npy', cube.astype(np.float32))
        result = self.classify(exe, '--input', self.dir / 'floats.npy')
        self.assertEqual(result.returncode, 1)
        self.assertIn("expected '<u2'", result.stderr)
        result = self.classify(exe, '--input', self.dir / 'cube.npy', cube_map=31 * cube.shape[1] * 240)
        self.assertEqual(result.returncode, 1)
        self.assertIn('too small for 32 rows', result.stderr)

    def test_own_cycle_count_added_up_over_the_strips(self):
        kernel = 'justounetsimple_opt_zynq7020'
        exe = self.build(kernel)
        cube, *_ = self.cube_and_labels(kernel)
        w = cube.shape[1]
        # 70 rows in strips of 32: 32, 32 and 6.
        result = self.classify(exe, '--input', self.dir / 'cube.npy', cube_map=64 * w * 240, scores_map=40 * w * 16,
                               cycles=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f'own count:          {3000 + 70 * w} cycles', result.stdout)
        result = self.classify(exe, '--input', self.dir / 'cube.npy')
        self.assertNotIn('own count', result.stdout)

    @unittest.skipUnless(shutil.which('sh') and shutil.which('awk'), 'no sh and awk')
    def test_benchmark_runs_every_kernel_on_every_image_and_writes_the_report(self):
        """software/opu/benchmark.sh on a fake board: each kernel's classify (one with its own
        count, one without), a kernel whose load.sh fails, two images."""
        top = self.dir / 'board'
        (top / 'firmware').mkdir(parents=True)
        shutil.copy2(OPU / 'benchmark.sh', top)
        for kernel, cycles in zip(KERNELS, (True, False)):
            folder = top / kernel
            folder.mkdir()
            shutil.copy2(self.build(kernel), folder / 'classify.bin')
            (folder / 'classify').write_text(f'#!/bin/sh\n{"FAKE_CYCLES=1 " if cycles else ""}'
                                             f'exec "$(dirname "$0")/classify.bin" "$@"\n')
            (folder / 'classify').chmod(0o755)
            (folder / 'load.sh').write_text('echo "loaded from $FIRMWARE"\n')
        broken = top / 'justoliunet_opt_zynq7020'
        broken.mkdir()
        (broken / 'classify').write_text('')
        (broken / 'load.sh').write_text('echo "md5sum: FAILED"; exit 1\n')
        # The same cube and labels for both images (the labels follow the first kernel's classes).
        _, labels, *_ = self.cube_and_labels(next(iter(KERNELS)))
        for image in ('one', 'two'):
            (top / image).mkdir()
            shutil.copy2(self.dir / 'cube.npy', top / image)
            shutil.copy2(self.dir / 'labels.npy', top / image)
        known = labels < 3
        want = {}
        for kernel in KERNELS:
            model = load_module(KERNELS[kernel], ROOT / 'tools' / f'{KERNELS[kernel]}.py')
            classes = model.with_class(model.classify_fixed(np.load(self.dir / 'cube.npy')))[..., 3]
            want[kernel] = 100 * (classes[known] == labels[known]).mean()
        env = {**os.environ, 'FAKE_CUBE_MAP': str(64 << 20), 'FAKE_SCORES_MAP': str(16 << 20), 'IMAGES': 'one two',
               'KERNELS': ' '.join(list(KERNELS) + ['justoliunet_opt_zynq7020', 'sp_unet_small_opt_zynq7020'])}
        result = subprocess.run(['sh', str(top / 'benchmark.sh')], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('skipped (no KERNEL/classify): sp_unet_small_opt_zynq7020', result.stdout)
        report = next((top / 'benchmark_results').glob('benchmark_*.md')).read_text()
        log = next((top / 'benchmark_results').glob('benchmark_*.log')).read_text()
        self.assertIn(f'loaded from {top / "firmware"}', log)
        rows = {line.split('|')[1].strip(): line for line in report.splitlines() if line.startswith('| `')}
        h, w = labels.shape
        counted, uncounted = KERNELS
        self.assertIn(f'| `{counted}` | 100.0 | 2 | {(1000 + h * w) / 6:.2f} a patch | {(1000 + h * w) / 1e5:.3f} |',
                      rows[f'`{counted}`'])
        self.assertIn(f'| `{uncounted}` | 100.0 | 2 | - | - |', rows[f'`{uncounted}`'])
        for kernel in KERNELS:
            self.assertIn(f'| {want[kernel]:.2f}% | {want[kernel]:.2f}% |', rows[f'`{kernel}`'])
        self.assertNotIn('`justoliunet_opt_zynq7020`', rows)
        self.assertEqual(report.count('failed: load.sh: md5sum: FAILED'), 2)
        self.assertEqual(report.count('| one | `'), 3)
        self.assertIn(f'one ({h} x {w}), two ({h} x {w})', report)
        csv_rows = next((top / 'benchmark_results').glob('benchmark_*.csv')).read_text().splitlines()
        self.assertEqual(len(csv_rows), 1 + 3 * 2)
        self.assertFalse((top / 'benchmark_results' / 'labels.bin').exists())


# The parts of an HLS driver header (xKERNEL.h) make_kernel_header.sh reads, as Vitis HLS
# writes them under __linux__.
DRIVER_HEADER = """\
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
typedef uint32_t u32;
typedef uint64_t u64;
typedef struct { u64 Control_BaseAddress; u32 IsReady; } XK_opt;
#define XST_SUCCESS 0
int XK_opt_Initialize(XK_opt *InstancePtr, const char* InstanceName);
int XK_opt_Release(XK_opt *InstancePtr);
void XK_opt_Start(XK_opt *InstancePtr);
u32 XK_opt_IsDone(XK_opt *InstancePtr);
u32 XK_opt_IsIdle(XK_opt *InstancePtr);
u32 XK_opt_IsReady(XK_opt *InstancePtr);
void XK_opt_Set_din10(XK_opt *InstancePtr, u64 Data);
void XK_opt_Set_din0(XK_opt *InstancePtr, u64 Data);
u64 XK_opt_Get_din0(XK_opt *InstancePtr);
void XK_opt_Set_din1(XK_opt *InstancePtr, u64 Data);
void XK_opt_Set_dout(XK_opt *InstancePtr, u64 Data);
void XK_opt_Set_height(XK_opt *InstancePtr, u32 Data);
void XK_opt_Set_width(XK_opt *InstancePtr, u32 Data);
CYCLES
#ifdef __cplusplus
}
#endif
"""
# The driver's functions, recording what classify's names call.
DRIVER_STUBS = r"""
#include "kernel.h"
#include <cstdio>
extern "C" {
int XK_opt_Initialize(XK_opt *, const char *name) { std::printf("init %s\n", name); return 0; }
int XK_opt_Release(XK_opt *) { return 0; }
void XK_opt_Start(XK_opt *) { std::printf("start\n"); }
u32 XK_opt_IsDone(XK_opt *) { return 1; }
u32 XK_opt_IsIdle(XK_opt *) { return 1; }
u32 XK_opt_IsReady(XK_opt *) { return 0; }
void XK_opt_Set_din0(XK_opt *, u64 a) { std::printf("din0 %llx\n", (unsigned long long)a); }
void XK_opt_Set_din1(XK_opt *, u64 a) { std::printf("din1 %llx\n", (unsigned long long)a); }
void XK_opt_Set_din10(XK_opt *, u64 a) { std::printf("din10 %llx\n", (unsigned long long)a); }
void XK_opt_Set_dout(XK_opt *, u64 a) { std::printf("dout %llx\n", (unsigned long long)a); }
void XK_opt_Set_height(XK_opt *, u32 v) { std::printf("height %u\n", v); }
void XK_opt_Set_width(XK_opt *, u32 v) { std::printf("width %u\n", v); }
CYCLES
}
int main() {
    KernelInstance k;
    kernelInitialize(&k, "hls_kernel");
    kernelSetInputs(&k, 0x20000000);
    kernelSetOutput(&k, 0x29600000);
    kernelSetSize(&k, 70, 45);
    kernelStart(&k);
    std::printf("done %d idle %d ready %d cycles %d %llu\n", kernelIsDone(&k), kernelIsIdle(&k), kernelIsReady(&k),
                kernelHasCycles(), (unsigned long long)kernelCycles(&k));
    std::printf("%s %.3f\n", KERNEL_NAME, KERNEL_CLOCK_MHZ);
}
"""


@unittest.skipIf(BASH is None, 'make_kernel_header.sh needs bash')
class KernelHeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, header):
        (self.dir / 'xk_opt.h').write_text(header)
        (self.dir / 'xk_opt_hw.h').write_text('#define XK_OPT_CONTROL_ADDR_AP_CTRL 0x00\n')
        return subprocess.run([BASH, str(OPU / 'make_kernel_header.sh'), str(self.dir), '78.125'],
                              capture_output=True, text=True)

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_classify_calls_the_kernels_own_driver(self):
        for cycles in (False, True):
            with self.subTest(cycles=cycles):
                declaration = 'u32 XK_opt_Get_cycles(XK_opt *InstancePtr);' if cycles else ''
                result = self.make(DRIVER_HEADER.replace('CYCLES', declaration))
                self.assertEqual(result.returncode, 0, result.stderr)
                (self.dir / 'kernel.h').write_text(result.stdout)
                stub = 'u32 XK_opt_Get_cycles(XK_opt *) { return 1234; }' if cycles else ''
                (self.dir / 'main.cpp').write_text(DRIVER_STUBS.replace('CYCLES', stub))
                subprocess.run([CXX, '-std=c++17', '-Wall', '-Wextra', '-Werror', f'-I{self.dir}',
                                str(self.dir / 'main.cpp'), '-o', str(self.dir / 'main')], check=True)
                out = subprocess.run([str(self.dir / 'main')], capture_output=True, text=True).stdout.splitlines()
                # Every input port reads the cube, in number order; then the output, the size, start.
                self.assertEqual(out[:8], ['init hls_kernel', 'din0 20000000', 'din1 20000000', 'din10 20000000',
                                           'dout 29600000', 'height 70', 'width 45', 'start'])
                self.assertEqual(out[8], f'done 1 idle 1 ready 0 cycles {int(cycles)} {1234 if cycles else 0}')
                self.assertEqual(out[9], 'k_opt 78.125')

    def test_a_kernel_without_the_shared_interface_is_refused(self):
        result = self.make(DRIVER_HEADER.replace('CYCLES', '').replace('Set_width', 'Set_n'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no XK_opt_Set_width', result.stderr)
        (self.dir / 'xk_opt.h').unlink()
        result = subprocess.run([BASH, str(OPU / 'make_kernel_header.sh'), str(self.dir), '100'],
                                capture_output=True, text=True)
        self.assertIn('no HLS driver header', result.stderr)


if __name__ == '__main__':
    unittest.main()
