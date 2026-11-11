"""software/opu/: cnn_accelerator, cnn-1d-picobob's program with this repository's kernels,
built on the PC from its real sources against a fake board, and run on a capture
(binned_cube.bip + capture_config.ini) bit-exact; make_kernel_header.sh (the kernel's HLS
driver under shared names); benchmark.sh; and tools/image.py score on what they write.

The fake board: src/uio_memory_device.cpp is replaced by one whose reserved memory is
plain memory, the HLS driver by one that runs the kernel's C++ when started, and
/proc/device-tree by a folder (FAKE_DEVICE_TREE) holding the overlay's class values.
Everything else is the program as it is built for the board. The first run fetches
spdlog, libconfig and argparse (as the board's build does) into build/opu-host-test/.
"""
import gc
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from helpers import CXX, ROOT, kernel_dir, load_module, raw_cube

try:
    import numpy as np
except ImportError:
    np = None

BASH = shutil.which('bash')
CMAKE = shutil.which('cmake')
OPU = ROOT / 'software/opu'
HOST_BUILD = ROOT / 'build/opu-host-test'
# Two kernels of different models, both the 7020's (two read ports), and their models.
KERNELS = {'justounetsimple_opt_zynq7020': 'justounetsimple_model', 'waveletcnn_opt_zynq7020': 'waveletcnn_model'}

# The reserved memory as plain memory: hls_kernel_memory with two maps, their sizes from
# FAKE_CUBE_MAP and FAKE_SCORES_MAP; the "physical" address is where it is. A start of the
# fake driver runs the kernel's C++ on it.
FAKE_UIO = r"""
#include "uio_memory_device.h"
#include "KERNEL.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

UioMap::UioMap(std::filesystem::path uio_device_path, std::filesystem::path map_dir)
    : device_path(uio_device_path) {
  readMapInfo(map_dir);
  buffer_ptr = std::calloc(map_size, 1);
  buffer_size = map_size;
  map_address = reinterpret_cast<uintptr_t>(buffer_ptr);
}
UioMap::~UioMap() { std::free(buffer_ptr); }
void UioMap::readMapInfo(std::filesystem::path map_dir) {
  map_index = std::stoi(map_dir.filename().string());
  map_name = map_index == 0 ? "cube_map" : "scores_map";
  map_size = std::strtoull(std::getenv(map_index == 0 ? "FAKE_CUBE_MAP" : "FAKE_SCORES_MAP"), nullptr, 0);
}
void UioMap::mapMemory(size_t size) {
  if (size > map_size) throw std::runtime_error("fake board: mapping more than the map");
}
void UioMap::unmapMemory() {}
void UioMap::dump(std::filesystem::path dump_path) {
  std::ofstream(dump_path, std::ios::binary).write(static_cast<char *>(buffer_ptr), buffer_size);
}
UioDevice::UioDevice(std::string name) : uio_name(name) {
  if (name != "hls_kernel_memory") throw std::runtime_error("No UIO device named " + name);
  n_maps = 2;
  maps_dir = "maps";
}
UioMap UioDevice::getMap(int map_index) { return UioMap("/dev/fake", maps_dir / std::to_string(map_index)); }

extern "C" uint64_t fake_kernel_run(uint64_t din0, uint64_t din1, uint64_t dout, uint32_t h, uint32_t w) {
  if (din0 != din1) { std::fprintf(stderr, "fake board: the read ports differ\n"); std::exit(3); }
  auto cube = reinterpret_cast<cube_word_t *>(din0);
  KERNEL(cube, cube, reinterpret_cast<score_word_t *>(dout), h, w);
  return 1000 + uint64_t(h) * w;                        // a count the tests can predict
}
"""

# The parts of the HLS driver (xKERNEL.h and .c) the program uses, as Vitis HLS writes them
# under __linux__; Initialize finds only hls_kernel.
FAKE_DRIVER_H = """\
#ifndef XKERNEL_H
#define XKERNEL_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#define XST_SUCCESS 0
#define XST_DEVICE_NOT_FOUND 2
#define XST_OPEN_DEVICE_FAILED 3
#define XIL_COMPONENT_IS_READY 0x11111111U
typedef struct { u64 din0, din1, dout, cycles; u32 height, width; } PREFIX;
int PREFIX_Initialize(PREFIX *InstancePtr, const char* InstanceName);
int PREFIX_Release(PREFIX *InstancePtr);
void PREFIX_Start(PREFIX *InstancePtr);
u32 PREFIX_IsDone(PREFIX *InstancePtr);
u32 PREFIX_IsIdle(PREFIX *InstancePtr);
u32 PREFIX_IsReady(PREFIX *InstancePtr);
void PREFIX_Set_din0(PREFIX *InstancePtr, u64 Data);
void PREFIX_Set_din1(PREFIX *InstancePtr, u64 Data);
void PREFIX_Set_dout(PREFIX *InstancePtr, u64 Data);
void PREFIX_Set_height(PREFIX *InstancePtr, u32 Data);
void PREFIX_Set_width(PREFIX *InstancePtr, u32 Data);
u64 PREFIX_Get_cycles(PREFIX *InstancePtr);
#ifdef __cplusplus
}
#endif
#endif
"""
FAKE_DRIVER_C = """\
#include "xKERNEL.h"
#include <string.h>
u64 fake_kernel_run(u64 din0, u64 din1, u64 dout, u32 h, u32 w);
int PREFIX_Initialize(PREFIX *k, const char *name) {
    memset(k, 0, sizeof *k);
    return strcmp(name, "hls_kernel") == 0 ? XST_SUCCESS : XST_DEVICE_NOT_FOUND;
}
int PREFIX_Release(PREFIX *k) { (void)k; return XST_SUCCESS; }
void PREFIX_Start(PREFIX *k) { k->cycles = fake_kernel_run(k->din0, k->din1, k->dout, k->height, k->width); }
u32 PREFIX_IsDone(PREFIX *k) { (void)k; return 1; }
u32 PREFIX_IsIdle(PREFIX *k) { (void)k; return 1; }
u32 PREFIX_IsReady(PREFIX *k) { (void)k; return 1; }
void PREFIX_Set_din0(PREFIX *k, u64 v) { k->din0 = v; }
void PREFIX_Set_din1(PREFIX *k, u64 v) { k->din1 = v; }
void PREFIX_Set_dout(PREFIX *k, u64 v) { k->dout = v; }
void PREFIX_Set_height(PREFIX *k, u32 v) { k->height = v; }
void PREFIX_Set_width(PREFIX *k, u32 v) { k->width = v; }
u64 PREFIX_Get_cycles(PREFIX *k) { return k->cycles; }
"""


def write_if_changed(path, text):
    """So that make rebuilds only what changed between runs."""
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


@unittest.skipIf(np is None, 'needs numpy')
@unittest.skipUnless(CXX and CMAKE and BASH, 'needs a C++ compiler, cmake and bash')
class AcceleratorTests(unittest.TestCase):
    built = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        gc.collect()
        self.tmp.cleanup()

    @classmethod
    def build(cls, kernel):
        """cnn_accelerator on the fake board running KERNEL, built once a test run (and only
        what changed since the last)."""
        if kernel in cls.built:
            return cls.built[kernel]
        app = HOST_BUILD / kernel / 'app'
        app.mkdir(parents=True, exist_ok=True)
        for item in ('CMakeLists.txt', 'cmake', 'src'):
            source = OPU / item
            if source.is_dir():
                shutil.copytree(source, app / item, dirs_exist_ok=True)
            else:
                shutil.copy2(source, app / item)
        # The fake board's parts in place of the board's.
        write_if_changed(app / 'src/uio_memory_device.cpp', FAKE_UIO.replace('KERNEL', kernel))
        xcnn = (OPU / 'src/xcnn_device.cpp').read_text()
        assert '"/proc/device-tree/"' in xcnn
        write_if_changed(app / 'src/xcnn_device.cpp', xcnn.replace('"/proc/device-tree/"', 'std::getenv("FAKE_DEVICE_TREE")'))
        driver = app / 'xcnn_driver'
        driver.mkdir(exist_ok=True)
        prefix = 'X' + kernel[0].upper() + kernel[1:]
        write_if_changed(driver / f'x{kernel}.h', FAKE_DRIVER_H.replace('PREFIX', prefix).replace('KERNEL', kernel.upper()))
        write_if_changed(driver / f'x{kernel}.c', FAKE_DRIVER_C.replace('PREFIX', prefix).replace('xKERNEL', f'x{kernel}'))
        header = subprocess.run([BASH, str(OPU / 'scripts/make_kernel_header.sh'), str(driver), '100.000'],
                                capture_output=True, text=True, check=True).stdout
        write_if_changed(driver / 'kernel.h', header)
        # The kernel's C++, compiled with the fake UIO.
        folder = kernel_dir(kernel)
        sources = ' '.join(f'#include "{cpp.name}"' for cpp in sorted(folder.glob('*.cpp')))
        write_if_changed(app / 'src/uio_memory_device.cpp', (app / 'src/uio_memory_device.cpp').read_text()
                         .replace('#include "KERNEL.hpp"'.replace('KERNEL', kernel), sources))
        out = HOST_BUILD / kernel / 'build'
        flags = f'-I{folder} -I{ROOT / "src/common"} -Wno-unknown-pragmas'
        subprocess.run([CMAKE, '-S', str(app), '-B', str(out), '-DCMAKE_BUILD_TYPE=Release',
                        f'-DCMAKE_CXX_FLAGS={flags}'], check=True, capture_output=True)
        result = subprocess.run([CMAKE, '--build', str(out), '-j', str(os.cpu_count() or 2)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        cls.built[kernel] = out / 'cnn_accelerator'
        return cls.built[kernel]

    def capture(self, kernel, folder=None, h=70, w=45):
        """A capture as HYPSO stores it, 3 x 2 patches with the edge ones partial, and
        labels.npy (mostly the kernel's classes; 3: unlabeled): the cube, the labels, and the
        scores the kernel must write."""
        folder = folder or self.dir
        folder.mkdir(parents=True, exist_ok=True)
        model = load_module(KERNELS[kernel], ROOT / 'tools' / f'{KERNELS[kernel]}.py')
        cube = raw_cube(model, h, w, seed=5, counts=True).astype(np.uint16)
        want = model.with_class(model.classify_fixed(cube))
        rng = np.random.default_rng(6)
        labels = np.where(rng.random((h, w)) < 0.8, want[..., 3], rng.integers(0, 4, (h, w)))
        cube.tofile(folder / 'binned_cube.bip')
        (folder / 'capture_config.ini').write_text(f'frame_count = {h}\nexposure = 47.32\nrow_count = {w}\naoi_y = 62\n')
        np.save(folder / 'labels.npy', labels)
        return cube, labels, want

    def device_tree(self, classes=True):
        """The overlay's node with the kernel's class values, as /proc/device-tree shows it."""
        node = self.dir / 'device-tree/amba_pl/hls_kernel/classes@0'
        node.mkdir(parents=True, exist_ok=True)
        (node / 'compatible').write_bytes(b'hypso,hls-kernel-classes\0')
        for name, value in (('sea', 2), ('land', 1), ('cloud', 0)):
            path = node / f'class-{name}'
            if classes:
                path.write_bytes(struct.pack('>I', value))
            elif path.exists():
                path.unlink()
        return self.dir / 'device-tree'

    def run_app(self, exe, folder=None, cube_map=64 << 20, scores_map=16 << 20, classes=True, **options):
        folder = folder or self.dir
        args = {'input': folder / 'binned_cube.bip', 'capture-config': folder / 'capture_config.ini',
                'output': self.dir / 'labels.bin', **options}
        env = {**os.environ, 'FAKE_CUBE_MAP': str(cube_map), 'FAKE_SCORES_MAP': str(scores_map),
               'FAKE_DEVICE_TREE': str(self.device_tree(classes))}
        command = [str(exe)] + [a for k, v in args.items() for a in (f'--{k}', str(v))]
        return subprocess.run(command, capture_output=True, text=True, env=env)

    def test_every_kernel_bit_exact_on_a_capture(self):
        for kernel in KERNELS:
            with self.subTest(kernel=kernel):
                exe = self.build(kernel)
                cube, _, want = self.capture(kernel)
                result = self.run_app(exe)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                # HYPSO's labels: sea 0, land 1, cloud 2 (the kernel's 2, 1, 0, from the devicetree).
                labels = np.fromfile(self.dir / 'labels.bin', dtype=np.uint8).reshape(cube.shape[:2])
                np.testing.assert_array_equal(labels, 2 - want[..., 3])
                self.assertIn(f'CnnDevice initialized for {kernel} (built for 100 MHz): width=45, height=70', result.stdout)
                self.assertIn('Labels are: sea=2, land=1, cloud=0', result.stdout)
                self.assertIn(f"CNN's own count: {1000 + 70 * 45} cycles", result.stdout)
                self.assertIn('Total runtime', result.stdout)

    def test_without_class_values_in_the_devicetree_the_labels_are_the_kernels(self):
        kernel = 'justounetsimple_opt_zynq7020'
        exe = self.build(kernel)
        cube, _, want = self.capture(kernel)
        result = self.run_app(exe, classes=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('using default label values', result.stdout)
        labels = np.fromfile(self.dir / 'labels.bin', dtype=np.uint8).reshape(cube.shape[:2])
        np.testing.assert_array_equal(labels, want[..., 3])

    def test_says_what_is_wrong(self):
        kernel = 'justounetsimple_opt_zynq7020'
        exe = self.build(kernel)
        cube, *_ = self.capture(kernel)
        h, w = cube.shape[:2]
        cases = (
            ({'cube_map': h * w * 240 - 1}, 'exceeds reserved memory size'),
            ({'scores_map': h * w * 16 - 1}, 'exceeds reserved memory size'),
        )
        for env, message in cases:
            with self.subTest(message=message, env=env):
                result = self.run_app(exe, **env)
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stdout)
        (self.dir / 'capture_config.ini').write_text(f'frame_count = {h + 1}\nexposure = 1\nrow_count = {w}\naoi_y = 0\n')
        result = self.run_app(exe)
        self.assertEqual(result.returncode, 1)
        self.assertIn('but it only contains', result.stdout)
        result = self.run_app(exe, input=self.dir / 'missing.bip')
        self.assertEqual(result.returncode, 1)
        self.assertIn('BIP input file not found', result.stdout)

    @unittest.skipUnless(shutil.which('sh') and shutil.which('awk'), 'no sh and awk')
    def test_benchmark_runs_every_kernel_on_every_image_and_the_pc_scores_them(self):
        """software/opu/scripts/benchmark.sh on a fake board: each kernel's cnn_accelerator, a
        kernel whose load.sh fails, two images; then tools/image.py score on the results."""
        top = self.dir / 'board'
        (top / 'firmware').mkdir(parents=True)
        shutil.copy2(OPU / 'scripts/benchmark.sh', top)
        tree = self.device_tree()
        for kernel in KERNELS:
            folder = top / kernel
            folder.mkdir()
            (folder / 'cnn_accelerator').write_text(f'#!/bin/sh\nexec "{self.build(kernel)}" "$@"\n')
            (folder / 'cnn_accelerator').chmod(0o755)
            (folder / 'load.sh').write_text('echo "loaded from $FIRMWARE"\n')
        broken = top / 'justoliunet_opt_zynq7020'
        broken.mkdir()
        (broken / 'cnn_accelerator').write_text('')
        (broken / 'load.sh').write_text('echo "md5sum: FAILED"; exit 1\n')
        first = next(iter(KERNELS))
        for image in ('one', 'two'):
            cube, labels, _ = self.capture(first, top / image)
        known = labels < 3
        want = {}
        for kernel in KERNELS:
            model = load_module(KERNELS[kernel], ROOT / 'tools' / f'{KERNELS[kernel]}.py')
            classes = model.with_class(model.classify_fixed(cube))[..., 3]
            want[kernel] = 100 * (classes[known] == labels[known]).mean()
        env = {**os.environ, 'FAKE_CUBE_MAP': str(64 << 20), 'FAKE_SCORES_MAP': str(16 << 20),
               'FAKE_DEVICE_TREE': str(tree), 'IMAGES': 'one two',
               'KERNELS': ' '.join(list(KERNELS) + ['justoliunet_opt_zynq7020', 'sp_unet_small_opt_zynq7020'])}
        result = subprocess.run(['sh', str(top / 'benchmark.sh')], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('skipped (no KERNEL/cnn_accelerator): sp_unet_small_opt_zynq7020', result.stdout)
        results = next((top / 'benchmark_results').iterdir())
        report = (results / 'benchmark.md').read_text()
        self.assertIn(f'loaded from {top / "firmware"}', (results / 'benchmark.log').read_text())
        rows = {line.split('|')[1].strip(): line for line in report.splitlines() if line.startswith('| `')}
        h, w = labels.shape
        for kernel in KERNELS:
            self.assertIn(f'| `{kernel}` | 100 | 2 | {(1000 + h * w) / 1e5:.3f} |', rows[f'`{kernel}`'])
        self.assertNotIn('`justoliunet_opt_zynq7020`', rows)
        self.assertEqual(report.count('failed: load.sh: md5sum: FAILED'), 2)
        self.assertEqual(len((results / 'benchmark.csv').read_text().splitlines()), 1 + 3 * 2)
        # The accuracy, on the PC, from the labels the board kept.
        score = subprocess.run([sys.executable, str(ROOT / 'tools/image.py'), 'score', str(results), '--images', str(top)],
                               capture_output=True, text=True)
        self.assertEqual(score.returncode, 0, score.stderr)
        for kernel in KERNELS:
            for image in ('one', 'two'):
                self.assertIn(f'| `{kernel}` | {image} | {want[kernel]:.2f}% |', score.stdout)
        one = subprocess.run([sys.executable, str(ROOT / 'tools/image.py'), 'score',
                              str(results / first / 'one-labels.bin'), str(top / 'one')], capture_output=True, text=True)
        self.assertIn(f'accuracy: {want[first]:.2f}% of {known.sum()} labelled pixels', one.stdout)


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
# The driver's functions, recording what kernel.h's names call.
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
        return subprocess.run([BASH, str(OPU / 'scripts/make_kernel_header.sh'), str(self.dir), '78.125'],
                              capture_output=True, text=True)

    @unittest.skipUnless(CXX, 'no C++ compiler')
    def test_kernel_h_calls_the_kernels_own_driver(self):
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
        result = subprocess.run([BASH, str(OPU / 'scripts/make_kernel_header.sh'), str(self.dir), '100'],
                                capture_output=True, text=True)
        self.assertIn('no HLS driver header', result.stderr)


if __name__ == '__main__':
    unittest.main()
