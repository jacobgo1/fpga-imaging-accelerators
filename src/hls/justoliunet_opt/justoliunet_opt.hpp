#ifndef JUSTOLIUNET_OPT_HLS_HPP
#define JUSTOLIUNET_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The quantized 1D-Justo-LiuNet (the same model as src/hls/justoliunet/, the
// golden baseline) on a whole raw capture: one start classifies every pixel.
// The ARM only points the kernel at the cube in DDR, as for justounetsimple_opt
// (same cube in, same scores out). justoliunet_bn_opt is the same with the
// BatchNorm model's weights.
//
//   read_pixels_ahead<0> ─┐  zscore  ┌─ conv1 ─ pool ─ conv2 ─ pool ─ conv3 ─ pool ─ conv4 ─ pool ─┐
//   read_pixels_ahead<1> ─┴─► 15 ────┘  17       17     16      16     18      18     16      16   └─ dense ─► store
//                                                                                                     16       1
//   cycles a pixel, each process; all of them run at once, a pixel each apart
//
// The image: a 1D network classifies each pixel from its own spectrum, so the
// image is one run of height x width pixels, read and written in memory order:
// no patches, no halos, no padding. Two readers take alternate blocks of 16
// pixels (240 words, one burst each), requested two blocks ahead.
//
// The work: 57 384 multiply-adds a pixel (only the conv outputs the pools keep:
// conv1 102 of 105, conv2 46 of 47), 9 x justounetsimple's, so this kernel is
// bound by its multipliers, not by DDR. Every layer gets the multipliers to do
// its share of a pixel in about the same time, ~16-18 cycles (C*_POS below: the
// output positions it computes at once, all channels and taps of each):
//
//   layer   MACs/pixel  per cycle            cycles  multipliers
//   conv1      3 672    6 pos x 6 ch x 6       17      216 x (12 x 8 bit)   LUTs
//   conv2     19 872    3 pos x 12 ch x 36     16      1 296, 2 per DSP     648 DSPs
//   conv3     23 328    1 pos x 18 ch x 72     18      1 296, 2 per DSP     648 DSPs
//   conv4     10 368    1 pos x 6 ch x 108     16      648 x (8 x 8 bit)    LUTs
//   dense        144    3 features x 3        16      9                    LUTs
//
// The DSPs go where the work is, conv2 and conv3 (75 % of it), two multiplies
// each (activations uint8 after ReLU, weights int8; conv3x3_packed.hpp's
// packing). conv1's 12-bit input is not packable and its layer small, conv4 is
// 18 % of the work: LUTs, so ~1 300 of the 1 728 DSPs are used and conv4 can
// move to DSPs (C4_PACKED) if LUTs run short. conv3 sets the pace: 18 cycles a
// pixel, about 59 ms for a 598 x 1092 capture at 200 MHz.
//
// Integer fixed point throughout: the 12-bit z-score (8 fraction bits) into
// conv1, uint8 activations after each block (fractions per model,
// justoliunet_opt_formats.hpp), int32 sums, int8 weights.
// tools/justoliunet_model.py has the bit-exact numpy version (classify_fixed);
// on the labelled aegean pixels it gives the same class as the trained fp32
// model on 99.7 % (justoliunet) and 99.1 % (justoliunet_bn) of them.
//
// din0, din1: the same raw cube, [height][width][120] uint16 (point both at it).
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
constexpr int BANDS = KEPT * PORT_BANDS;        // 112 (110 and 111 are zero)
constexpr int K = 6;             // every conv's kernel
constexpr int CLASSES = 3;
constexpr int PORTS = 2;         // read ports, alternate blocks
constexpr int BLOCK = 16;        // pixels a burst: 240 words

// The layers' parallelism and where their multipliers go (see above).
constexpr int C1_POS = 6;   constexpr bool C1_DSP = false;
constexpr int C2_POS = 3;   constexpr bool C2_PACKED = true;
constexpr int C3_POS = 1;   constexpr bool C3_PACKED = true;
constexpr int C4_POS = 1;   constexpr int C4_OCP = 6;   constexpr bool C4_PACKED = false;
constexpr int DENSE_FEAT = 3;
constexpr bool ZSCORE_DSP = true;   // 8 multipliers, 8 DSPs

// One word of the cube: 8 bands of a pixel, 128 bits.
typedef hls::vector<uint16_t, PORT_BANDS> jl_raw_t;
// One pixel of dout: 3 scores and the class, 128 bits.
typedef hls::vector<int32_t, 4> jl_out_t;

void justoliunet_opt(hls::burst_maxi<jl_raw_t> din0, hls::burst_maxi<jl_raw_t> din1,
                     hls::burst_maxi<jl_out_t> dout, int height, int width, uint32_t& cycles);

#endif // JUSTOLIUNET_OPT_HLS_HPP
