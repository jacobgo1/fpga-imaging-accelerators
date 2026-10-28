#ifndef SPECTRALPOINTWISE1D_OPT_HLS_HPP
#define SPECTRALPOINTWISE1D_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The quantized 1D-SpectralPointwise (the same model as src/hls/spectralpointwise1d/,
// the golden baseline) on a whole raw capture: one start classifies every pixel.
// The ARM only points the kernel at the cube in DDR, as for justoliunet_opt (same
// cube in, same scores out, the same notebook and OPU program pattern).
//
//   read_pixels_ahead<0> ─► zscore + convsp1 ─► convsp2 ─► convsp3 ─┐  lane 0 (HP0)
//   read_pixels_ahead<1> ─► zscore + convsp1 ─► convsp2 ─► convsp3 ─┤  lane 1 (HP1)
//   read_pixels_ahead<2> ─► zscore + convsp1 ─► convsp2 ─► convsp3 ─┼─► store (HPC0)
//   read_pixels_ahead<3> ─► zscore + convsp1 ─► convsp2 ─► convsp3 ─┘  lane 3 (HP3)
//            15                    15              12          3        1
//   cycles a pixel of its lane, each process; all of them run at once
//
// The work is small: 2 964 multiply-adds a pixel (convsp1 110 x 24 = 2 640, convsp2
// 288, convsp3 36), 19x fewer than 1D-Justo-LiuNet, against 240 bytes of cube read.
// So this kernel is bound by DDR, and is built around the readers: the whole
// network is copied once per read port (a lane), each copy classifying its
// reader's blocks of 16 pixels (lane r: blocks r, r + 4, r + 8, ...), and the
// store takes the lanes' scores in turn, back in pixel order (pixel_io.hpp).
// Four lanes on HP0-3, the scores through HPC0: a pixel every 15 / 4 = 3.75
// cycles, 13 ms for a 598 x 1092 capture at 187.5 MHz, if DDR gives the four
// readers the 12 GB/s that needs (justounetsimple_opt needs 11.2; what it gives
// is the limit, not the compute).
//
// In a lane, convsp1 runs on the raw words as they arrive (zscore_pointwise.hpp):
// each cycle one word of 8 bands, z-scored and multiplied into all 24 sums, so a
// pixel takes its reader's 15 words and no spectrum is ever assembled. convsp2 and
// convsp3 take a whole pixel's channels at once, one output channel a cycle
// (pointwise1d.hpp), well inside the 15:
//
//   layer     MACs/pixel  per cycle          cycles  multipliers, a lane
//   z-score       110     8 bands               15     8 (16 x 17 bit)      DSPs
//   convsp1     2 640     8 bands x 24 ch       15   192 (12 x 8 bit)       DSPs
//   convsp2       288     1 ch x 24             12    24 (8 x 8 bit)        DSPs
//   convsp3        36     1 class x 12           3    12                    LUTs
//
// 224 DSPs a lane, 896 of the 1 728. convsp1's 12-bit input cannot share a DSP
// between two products (conv1d_engine.hpp), and there is no need to: DSPs are
// plenty here.
//
// Integer fixed point throughout: 1D-Justo-LiuNet's 12-bit z-score (8 fraction
// bits, justoliunet_opt_formats.hpp) into convsp1, uint8 activations after convsp1
// and convsp2 (spectralpointwise1d_opt_formats.hpp), int32 sums, int8 weights.
// tools/spectralpointwise1d_model.py has the bit-exact numpy version
// (classify_fixed); on the labelled aegean pixels it gives the same class as the
// trained fp32 model on 99.7 % of them.
//
// din0 .. din3: the same raw cube, [height][width][120] uint16 (point all at it).
// dout: [height][width][4] int32: the 3 scores (score * 2^16), then the class
// (0 cloud, 1 land, 2 sea; the first highest score).
// cycles: an output register, the kernel's time in clock cycles, from its start
// until the last score is in memory. 0 in C simulation.
// Any height and width from 1 to 65535; a strip of a larger image works the
// same way.

constexpr int RAW_BANDS = 120;   // bands in the cube
constexpr int PORT_BANDS = 8;    // bands per 128-bit word
constexpr int WORDS = RAW_BANDS / PORT_BANDS;   // 15 words a pixel
constexpr int FIRST = 1;         // word 0 (raw bands 0-7) is dropped
constexpr int KEPT = WORDS - FIRST;             // words 1-14: kept bands 0-111
constexpr int CLASSES = 3;
constexpr int LANES = 4;         // read ports, each with its own copy of the network
constexpr int BLOCK = 16;        // pixels a burst: 240 words

// Where the multipliers go, and the output channels per cycle (see above).
constexpr bool ZSCORE_DSP = true;
constexpr bool SP1_DSP = true;
constexpr int SP2_PO = 1;   constexpr bool SP2_DSP = true;
constexpr int SP3_PO = 1;   constexpr bool SP3_DSP = false;

// One word of the cube: 8 bands of a pixel, 128 bits.
typedef hls::vector<uint16_t, PORT_BANDS> spw_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> spw_out_t;

void spectralpointwise1d_opt(hls::burst_maxi<spw_raw_t> din0, hls::burst_maxi<spw_raw_t> din1,
                             hls::burst_maxi<spw_raw_t> din2, hls::burst_maxi<spw_raw_t> din3,
                             hls::burst_maxi<spw_out_t> dout, int height, int width, uint32_t& cycles);

#endif // SPECTRALPOINTWISE1D_OPT_HLS_HPP
