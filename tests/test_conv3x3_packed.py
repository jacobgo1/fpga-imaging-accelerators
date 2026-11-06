"""conv3x3_tiled's packed mode (two multiplies per DSP, src/optimized/dsp_pack.hpp), with plain g++.

The networks' weights are small, so their testbenches never come near the bound a group of
products must stay within: a group of 8 uint8 products (the int8 bound) passed
sp_unet_small_opt_u8's testbench. Here the weights are the worst case, -128 and 127, and the
activations near 255, so a group one product too long overflows the low channel. Against a
plain loop, every sum, with whole groups (IN_PER_CYCLE 4: 36 taps) and a shorter last one
(1 and 2: 9 and 18 taps), two and four outputs a step; and the unpacked path on uint8.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers import CXX, ROOT, compile_cpp

DRIVER = """
#include "conv3x3_tiled.hpp"
#include <cstdio>
#include <cstdlib>

constexpr int H = 8, W = 8, IN = 4, OUT = 4;
const weight_t WEIGHT[OUT][IN][3][3] = {{{weights}}};
const weight_t BIAS[OUT] = {{3, -5, 7, -11}};

int main() {{
    static uact_t x[H][W][IN];
    unsigned seed = 7;
    for (int y = 0; y < H; y++)
        for (int c = 0; c < W; c++)
            for (int i = 0; i < IN; i++) {{
                seed = seed * 1103515245u + 12345u;
                x[y][c][i] = (uact_t)(seed >> 16) % 8 == 0 ? (uact_t)((seed >> 8) & 0xff) : (uact_t)255;
            }}
    hls::stream<vec_t<uact_t, IN> > in;
    hls::stream<vec_t<sum_t, OUT> > out;
    for (int y = 0; y < H; y++)
        for (int c = 0; c < W; c++) {{
            vec_t<uact_t, IN> v;
            for (int i = 0; i < IN; i++) v.v[i] = x[y][c][i];
            in.write(v);
        }}
    conv3x3_tiled<H, W, H, W, IN, OUT, {ipc}, {opc}, 0, WEIGHT, BIAS, true, activation::none, 0, {packed}, 18>(
        in, out, H, W);
    int errors = 0;
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
                                want += (long long)x[yy][xx][i] * WEIGHT[o][i][ky][kx];
                        }}
                if (got.v[o] != want) errors++;
            }}
        }}
    std::printf("%d\\n", errors);
    return 0;
}}
"""


def weights():
    """Worst case: every weight -128 or 127, mostly -128 (the largest product)."""
    values = [-128 if (k * 7) % 5 else 127 for k in range(4 * 4 * 9)]
    return ', '.join(str(v) for v in values)


@unittest.skipUnless(CXX, 'no C++ compiler')
class PackedConvTests(unittest.TestCase):
    def run_conv(self, tmp, ipc, opc, packed):
        stem = f'conv_{ipc}_{opc}_{int(packed)}'
        exe = compile_cpp(tmp, stem, DRIVER.format(weights=weights(), ipc=ipc, opc=opc,
                                                    packed='true' if packed else 'false'),
                          [ROOT / 'src/optimized'])
        return int(subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout)

    def test_packed_sums_are_exact_with_worst_case_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ipc in (1, 2, 4):
                for opc in (2, 4):
                    with self.subTest(in_per_cycle=ipc, out_per_cycle=opc):
                        self.assertEqual(self.run_conv(Path(tmp), ipc, opc, True), 0)

    def test_unpacked_uint8_sums_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_conv(Path(tmp), 2, 2, False), 0)


if __name__ == '__main__':
    unittest.main()
