#ifndef SPECTRALPOINTWISE1D_OPT_ZYNQ7030_HLS_HPP
#define SPECTRALPOINTWISE1D_OPT_ZYNQ7030_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// spectralpointwise1d_opt (read its header first) for the Zynq-7030, HYPSO-2's
// FPGA: build it with --board zynq7030. The same model, the same integer formats,
// the same answers bit for bit, the same raw cube in and scores out, and the same
// four lanes (spectralpointwise1d_opt_dataflow.hpp); what changes is the ports:
//
//   read_pixels_ahead<0..3> ─► zscore + convsp1 ─► convsp2 ─► convsp3 ─► store
//          HP0-3                    30                12          3      (shares HP3)
//   cycles a pixel of its lane, each process; four lanes, a pixel every 7.5 cycles
//
// - 64-bit HP ports: a word is 4 bands, a pixel 30 words, so a lane takes 30
//   cycles a pixel; four lanes on HP0-3 (the writer shares HP3, as
//   justounetsimple_opt_zynq7030's), one pixel every 7.5 cycles: 49 ms for a
//   598 x 1092 capture at 100 MHz, if DDR gives the four readers the 3.2 GB/s
//   that needs (justounetsimple_opt_zynq7030 needs 2.4; check on HYPSO-2).
// - Each word's 4 bands go into all 24 of convsp1's sums: 96 multipliers (12 x 8
//   bit) a lane, in DSPs, 384 of the 400. The rest is in LUTs: the z-score's 4
//   multipliers (16 x 17 bit), convsp2's 24 and convsp3's 12 a lane.
//
//   layer     MACs/pixel  per cycle          cycles  multipliers, a lane
//   z-score       110     4 bands               30     4                    LUTs
//   convsp1     2 640     4 bands x 24 ch       30    96 (12 x 8 bit)       DSPs
//   convsp2       288     1 ch x 24             12    24 (8 x 8 bit)        LUTs
//   convsp3        36     1 class x 12           3    12                    LUTs
//
// - Blocks of 8 pixels, 240 words a burst.
//
// din0 .. din3: the raw cube, [height][width][120] uint16 (point all at it).
// dout: [height][width][4] int32: the 3 scores (score * 2^16), then the class (0
// cloud, 1 land, 2 sea). cycles: the kernel's own count of its clock cycles. Any
// height and width from 1 to 65535.

constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int PORT_BANDS = 4;    // bands per 64-bit word
constexpr int WORDS = RAW_BANDS / PORT_BANDS;   // 30 words a pixel
constexpr int FIRST = 2;         // words 0-1 (raw bands 0-7) are dropped
constexpr int KEPT = WORDS - FIRST;             // words 2-29: kept bands 0-111
constexpr int CLASSES = 3;
constexpr int LANES = 4;         // read ports, each with its own copy of the network
constexpr int BLOCK = 8;         // pixels a burst: 240 words

// Where the multipliers go, and the output channels per cycle (see above).
constexpr bool ZSCORE_DSP = false;
constexpr bool SP1_DSP = true;
constexpr int SP2_PO = 1;   constexpr bool SP2_DSP = false;
constexpr int SP3_PO = 1;   constexpr bool SP3_DSP = false;

// One word of the cube: 4 bands of a pixel, 64 bits.
typedef hls::vector<uint16_t, PORT_BANDS> spw_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> spw_out_t;

void spectralpointwise1d_opt_zynq7030(hls::burst_maxi<spw_raw_t> din0, hls::burst_maxi<spw_raw_t> din1,
                                      hls::burst_maxi<spw_raw_t> din2, hls::burst_maxi<spw_raw_t> din3,
                                      hls::burst_maxi<spw_out_t> dout, int height, int width, uint32_t& cycles);

#endif // SPECTRALPOINTWISE1D_OPT_ZYNQ7030_HLS_HPP
