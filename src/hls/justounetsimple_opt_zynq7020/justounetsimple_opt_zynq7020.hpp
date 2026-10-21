#ifndef JUSTOUNETSIMPLE_OPT_ZYNQ7020_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_ZYNQ7020_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// justounetsimple_opt (read its header first) sized for the Zynq-7020 (the
// PYNQ-Z2 in the lab): build it with --board zynq7020. The same model, the same
// integer arithmetic, the same answers bit for bit, the same raw cube in and
// scores out; what changes is what fits on the chip (boards/README.md):
//
//   read_rows_ahead<0> ─┬─► patch_buffer ─► conv1 ─► pool ─► conv2 ─► pool+up ─►
//   read_rows_ahead<1> ─┘   z-score, BRAM          conv3 ─► up ─► conv4 ─► store (HP2)
//
// - 220 DSPs, and conv1 takes 216 of them: 8 bands per cycle, two multiplies
//   per DSP (zynq7020/conv3x3_wide.hpp: the 7-series DSP48E1 is 2 bits
//   narrower than the ZCU104's, so the packing differs), 3 channel pairs x 8 x
//   9 = 216 DSPs, 14 x 32 x 32 + 33 = 14 369 cycles per patch.
// - Every other multiply is in LUTs (DSP = false in the .cpp):
//   conv2-4 do 2 input x one output channel per cycle (conv3x3_serial, 18
//   multipliers each) and take 9 828 cycles a patch each for conv2-3 (9 483 for
//   conv4); the z-score's 8 multipliers; the address arithmetic.
// - 64-bit HP ports: a word of the cube is 4 bands, so a pixel is 30 words.
//   Two read ports (HP0, HP1) each take every other row of a patch in 4 bursts
//   of 8 pixels (240 words, pipelining the next burst while the current one
//   streams): 15 360 cycles a patch, the slowest stage (7% over conv1). The
//   scores go out through HP2.
// - No UltraRAM: the patch buffer is in block RAM (56 of 140 BRAM36).
// At 100 MHz that is about 0.16 ms per patch (0.10 on the 7030, which does 12
// bands a cycle with its 400 DSPs; 0.04 on the ZCU104).
//
// din0, din1: the same raw cube, [height][width][120] uint16. The 120 bands
// of a pixel are 30 words of 4; words 2-29 are the kept bands (raw 8-117)
// plus raw 118-119, which the z-score zeroes: two words per cycle of conv1.
// Words 0-1 (raw 0-7) are dropped.
// dout: [height][width][4] int32, as justounetsimple_opt's: the 3 scores
// (score * 2^OUT_FRAC), then the class (0 cloud, 1 land, 2 sea).
// Any height and width from 1 to 65535 (the index arithmetic is 16-bit).

// The kernel's sizes and formats, used as they are by the .cpp and the testbench.
// constexpr, not #define: a macro H would also rewrite every template
// parameter called H in src/optimized/; a constant is just hidden by it.
constexpr int H = 32, W = 32;    // patch size
constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int IN_CH = 110;       // bands the model uses: raw 8-117
constexpr int BASE_CH = 6;
constexpr int OUT_CH = 3;
constexpr int PORTS = 2;         // read ports, each half of the rows
constexpr int P1 = 8;            // bands per cycle into conv1
constexpr int P2 = 2;            // input channels per cycle in conv2-4
constexpr int PORT_BANDS = 4;    // bands per 64-bit word
// IN_CH rounded up to a multiple of P1: 112 bands, the last 2 zero.
constexpr int IN_PAD = (IN_CH + P1 - 1) / P1 * P1;

constexpr int IN_FRAC = 5;       // conv1's input: z-score * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 11;     // activations between the blocks (int16: -16 .. +16)
constexpr int OUT_FRAC = 16;     // dout = score * 2^16

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> jopt_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> jopt_out_t;

void justounetsimple_opt_zynq7020(hls::burst_maxi<jopt_raw_t> din0, hls::burst_maxi<jopt_raw_t> din1,
                                  hls::burst_maxi<jopt_out_t> dout, int height, int width);

#endif // JUSTOUNETSIMPLE_OPT_ZYNQ7020_HLS_HPP
