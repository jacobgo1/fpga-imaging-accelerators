"""conv3x3_tiled_chunks (a pixel in chunks, two multiplies per DSP), with plain g++.

justounetsimple_opt_tiled's testbench cannot fail some of its bugs: the network's weights never
come near a group's bound, the padded bands always arrive as zeros, and 24 bands a chunk make
216 taps, whole groups of 8. Here the worst case: weights -128 and 127, int8 inputs mostly 127
(or uint8 255), 10 input channels in chunks of 3, 4 and 8 (padding of 2, 2 and 6 channels, with
junk in them, and a shorter last group), one and two output groups, two patches. Against a
plain loop, every sum.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers import CXX, ROOT, compile_cpp

DRIVER = """
#include "conv3x3_tiled_chunks.hpp"
#include <cstdio>

typedef {in_t} in_t;
constexpr int H = 8, W = 8, PATCHES = 2, IN = 10, OUT = 4, IPC = {ipc}, OPC = {opc};
constexpr int CHUNKS = (IN + IPC - 1) / IPC;
const weight_t WEIGHT[OUT][IN][3][3] = {{{weights}}};
const weight_t BIAS[OUT] = {{3, -5, 7, -11}};

int main() {{
    static in_t x[PATCHES][H][W][IN];
    hls::stream<vec_t<in_t, IPC> > in;
    hls::stream<vec_t<sum_t, OUT> > out;
    unsigned seed = 7;
    for (int k = 0; k < PATCHES; k++)
        for (int y = 0; y < H; y++)
            for (int c = 0; c < W; c++) {{
                for (int i = 0; i < IN; i++) {{
                    seed = seed * 1103515245u + 12345u;
                    const int r = (int)((seed >> 8) & 0xff) - 128;
                    x[k][y][c][i] = (seed >> 16) % 8 == 0 ? (in_t)(r < -127 ? -127 : r) : (in_t){big};
                }}
                for (int g = 0; g < CHUNKS; g++) {{
                    vec_t<in_t, IPC> v;
                    for (int i = 0; i < IPC; i++)
                        v.v[i] = g * IPC + i < IN ? x[k][y][c][g * IPC + i] : (in_t){big};   // junk past IN
                    in.write(v);
                }}
            }}
    conv3x3_tiled_chunks<H, W, H, W, IN, OUT, IPC, OPC, 0, WEIGHT, BIAS, 18, activation::none, 0>(
        in, out, PATCHES * H, W);
    int errors = 0;
    for (int k = 0; k < PATCHES; k++)
        for (int y = 0; y < H; y++)
            for (int c = 0; c < W; c++) {{
                const vec_t<sum_t, OUT> got = out.read();
                for (int o = 0; o < OUT; o++) {{
                    long long want = BIAS[o];
                    for (int i = 0; i < IN; i++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++) {{
                                const int yy = y + ky - 1, xx = c + kx - 1;
                                if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                                    want += (long long)x[k][yy][xx][i] * WEIGHT[o][i][ky][kx];
                            }}
                    if (got.v[o] != want) errors++;
                }}
            }}
    std::printf("%d\\n", errors);
    return 0;
}}
"""


def weights():
    """Worst case: every weight -128 or 127, mostly -128 (the largest product); a pattern of
    11, so that no two output channels (90 weights each) have the same."""
    values = [-128 if (k * 7) % 11 < 8 else 127 for k in range(4 * 10 * 9)]
    return ', '.join(str(v) for v in values)


@unittest.skipUnless(CXX, 'no C++ compiler')
class ChunkedConvTests(unittest.TestCase):
    def run_conv(self, tmp, ipc, opc, in_t):
        stem = f'chunks_{ipc}_{opc}_{in_t}'
        big = '127' if in_t == 'act8_t' else '255'
        exe = compile_cpp(tmp, stem, DRIVER.format(weights=weights(), ipc=ipc, opc=opc, in_t=in_t, big=big),
                          [ROOT / 'src/optimized'])
        return int(subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout)

    def test_int8_sums_are_exact_with_worst_case_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ipc in (3, 4, 8):
                for opc in (2, 4):
                    with self.subTest(in_per_cycle=ipc, out_per_cycle=opc):
                        self.assertEqual(self.run_conv(Path(tmp), ipc, opc, 'act8_t'), 0)

    def test_uint8_sums_are_exact_with_worst_case_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ipc in (3, 4):
                with self.subTest(in_per_cycle=ipc):
                    self.assertEqual(self.run_conv(Path(tmp), ipc, 2, 'uact_t'), 0)


if __name__ == '__main__':
    unittest.main()
