#ifndef JUSTOUNETSIMPLE_OPT_ZYNQ7030_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_ZYNQ7030_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// justounetsimple_opt (read its header first) sized for the Zynq-7030, HYPSO-2's
// FPGA: build it with --board zynq7030. The same model, the same integer
// arithmetic, the same answers bit for bit, the same raw cube in and scores
// out. It is the Zynq-7020 version (justounetsimple_opt_zynq7020.hpp) made
// wider for the 7030's larger fabric (400 DSPs against 220, 79 k LUTs against
// 53 k, 265 BRAM36 against 140) and 100 MHz clock:
//
//   read_rows_ahead<0..3> ─► patch_buffer ─► conv1 ─► pool ─► conv2 ─► pool+up ─►
//   HP0-HP3 (4 ports)        z-score, BRAM          conv3 ─► up ─► conv4 ─► store (HP3)
//
// - conv1 takes 324 of the 400 DSPs: 12 bands per cycle, two multiplies per
//   DSP (packed as on the 7020: the same DSP48E1), 3 channel pairs x 12 x 9 =
//   324 DSPs, 10 x 32 x 32 + 33 = 10 273 cycles per patch (7020: 8 bands, 216
//   DSPs, 14 369). 16 bands a cycle would need 432.
// - conv2-4 as on the 7020 (2 input x one output channel per cycle,
//   conv3x3_serial, 9 828 cycles a patch each for conv2-3, 9 483 for conv4),
//   but on DSPs: 18 each, 54 of the 76 left. The z-score's 16 multipliers are
//   in LUTs.
// - Four read ports (HP0-HP3), each every fourth row of a patch in 4 bursts of
//   8 pixels (240 words, pipelining the next burst while the current one
//   streams): 7 680 cycles a patch, under conv1's (with two, as on the 7020,
//   15 360 would set the pace). The scores' writer shares HP3 with the fourth
//   reader (boards/zynq7030/system.tcl): AXI reads and writes have their own
//   channels.
// - The patch buffer in block RAM, 3 words a chunk: about 60 of 265 BRAM36.
// conv1 sets the pace: 10 273 cycles a patch, 0.10 ms at 100 MHz, about 68 ms
// for a 598 x 1092 capture (7020: 15 360 at 78.1 MHz, 131 ms).
//
// din0 .. din3: the same raw cube, [height][width][120] uint16. The 120 bands
// of a pixel are 30 words of 4; words 2-29 are the kept bands (raw 8-117)
// plus raw 118-119, which the z-score zeroes: three words per cycle of conv1
// (its last chunk has one word past them, zero).
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
constexpr int PORTS = 4;         // read ports, each a quarter of the rows
constexpr int P1 = 12;           // bands per cycle into conv1
constexpr int P2 = 2;            // input channels per cycle in conv2-4
constexpr int PORT_BANDS = 4;    // bands per 64-bit word
// IN_CH rounded up to a multiple of P1: 120 bands, the last 10 zero.
constexpr int IN_PAD = (IN_CH + P1 - 1) / P1 * P1;

constexpr int IN_FRAC = 5;       // conv1's input: z-score * 2^5 (int8: -3.97 .. +3.97)
constexpr int ACT_FRAC = 11;     // activations between the blocks (int16: -16 .. +16)
constexpr int OUT_FRAC = 16;     // dout = score * 2^16

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> jopt_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> jopt_out_t;

void justounetsimple_opt_zynq7030(hls::burst_maxi<jopt_raw_t> din0, hls::burst_maxi<jopt_raw_t> din1,
                                  hls::burst_maxi<jopt_raw_t> din2, hls::burst_maxi<jopt_raw_t> din3,
                                  hls::burst_maxi<jopt_out_t> dout, int height, int width);

#endif // JUSTOUNETSIMPLE_OPT_ZYNQ7030_HLS_HPP
