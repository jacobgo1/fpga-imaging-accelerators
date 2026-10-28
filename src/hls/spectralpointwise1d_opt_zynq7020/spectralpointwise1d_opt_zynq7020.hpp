#ifndef SPECTRALPOINTWISE1D_OPT_ZYNQ7020_HLS_HPP
#define SPECTRALPOINTWISE1D_OPT_ZYNQ7020_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// spectralpointwise1d_opt (read its header first) for the Zynq-7020, the lab's
// ZedBoards: build it with --board zedboard (or zynq7020 for the PYNQ-Z2). The
// same model, the same integer formats, the same answers bit for bit, the same
// raw cube in and scores out, the same lanes (spectralpointwise1d_opt_lanes.hpp),
// two of them:
//
//   read_pixels_ahead<0> ─► zscore + convsp1 ─► convsp2 ─► convsp3 ─┐  lane 0 (HP0)
//   read_pixels_ahead<1> ─► zscore + convsp1 ─► convsp2 ─► convsp3 ─┴─► store (HP2)
//                                   30               12          3      lane 1 (HP1)
//   cycles a pixel of its lane, each process; a pixel every 15 cycles
//
// - Two lanes, as justounetsimple_opt_zynq7020 has two read ports: a lane takes a
//   pixel's 30 words of 4 bands (64-bit ports) in 30 cycles, two lanes a pixel
//   every 15 cycles: 125 ms for a 598 x 1092 capture at 78.1 MHz, if DDR gives
//   the two readers the 1.25 GB/s that needs (as justounetsimple_opt_zynq7020).
// - The DSPs: convsp1's 96 multipliers (12 x 8 bit) and the z-score's 4 (16 x 17
//   bit) a lane, 200 of the 220; convsp2 and convsp3 in LUTs.
//
//   layer     MACs/pixel  per cycle          cycles  multipliers, a lane
//   z-score       110     4 bands               30     4                    DSPs
//   convsp1     2 640     4 bands x 24 ch       30    96 (12 x 8 bit)       DSPs
//   convsp2       288     1 ch x 24             12    24 (8 x 8 bit)        LUTs
//   convsp3        36     1 class x 12           3    12                    LUTs
//
// - Blocks of 8 pixels, 240 words a burst.
//
// din0, din1: the raw cube, [height][width][120] uint16 (point both at it).
// dout: [height][width][4] int32: the 3 scores (score * 2^16), then the class (0
// cloud, 1 land, 2 sea). cycles: the kernel's own count of its clock cycles. Any
// height and width from 1 to 65535.

constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int PORT_BANDS = 4;    // bands per 64-bit word
constexpr int WORDS = RAW_BANDS / PORT_BANDS;   // 30 words a pixel
constexpr int FIRST = 2;         // words 0-1 (raw bands 0-7) are dropped
constexpr int KEPT = WORDS - FIRST;             // words 2-29: kept bands 0-111
constexpr int CLASSES = 3;
constexpr int LANES = 2;         // read ports, each with its own copy of the network
constexpr int BLOCK = 8;         // pixels a burst: 240 words

// Where the multipliers go, and the output channels per cycle (see above).
constexpr bool ZSCORE_DSP = true;
constexpr bool SP1_DSP = true;
constexpr int SP2_PO = 1;   constexpr bool SP2_DSP = false;
constexpr int SP3_PO = 1;   constexpr bool SP3_DSP = false;

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> spw_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> spw_out_t;

void spectralpointwise1d_opt_zynq7020(hls::burst_maxi<spw_raw_t> din0, hls::burst_maxi<spw_raw_t> din1,
                                      hls::burst_maxi<spw_out_t> dout, int height, int width, uint32_t& cycles);

#endif // SPECTRALPOINTWISE1D_OPT_ZYNQ7020_HLS_HPP
