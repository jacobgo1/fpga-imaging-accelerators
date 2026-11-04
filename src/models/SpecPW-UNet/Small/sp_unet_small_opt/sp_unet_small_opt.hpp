#ifndef SP_UNET_SMALL_OPT_DIRECT_HLS_HPP
#define SP_UNET_SMALL_OPT_DIRECT_HLS_HPP

#include "sp_unet_opt.hpp"

// SpecPW-UNet small, direct convs, on the ZCU104: the chain and its sizes are in the .cpp.
// Every process takes at most T = 9 828 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 16;     // convsp1: 16 bands a cycle, 7 cycles a pixel (7 168), 192 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs: in LUTs they took ~25 k LUTs (small)
constexpr int CONVSP2_OUT_PER_CYCLE = 2;        // convsp2 24 -> 12: 6 cycles a pixel (6 144), 48 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 2;        // convsp3 12 -> 12: 6 (6 144), 24 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 8 -> 3: 3 (3 072), 8 in LUTs
// The convs: channels, cycles a pixel x pixels (+ a row) = cycles a patch, multipliers.
typedef layer_size< 4, 4, true> DOUBLE1_CONV1;  // 12 -> 12   3 x 3 = 9      x 1 057 = 9 513      144
typedef layer_size< 4, 4, true> DOUBLE1_CONV2;  // 12 -> 12   3 x 3 = 9      x 1 057 = 9 513      144
typedef layer_size< 4, 2, true> DOUBLE2_CONV1;  // 12 -> 24   3 x 12 = 36    x 273 = 9 828         72
typedef layer_size< 8, 2, true> DOUBLE2_CONV2;  // 24 -> 24   3 x 12 = 36    x 273 = 9 828        144
typedef layer_size< 8, 2, true> UP1_CONV;       // 24 -> 24   3 x 12 = 36    x 273 = 9 828        144
typedef layer_size< 8, 1, true> DOUBLE3_CONV1;  // 24 -> 12   3 x 12 = 36    x 273 = 9 828         72
typedef layer_size< 4, 1, true> DOUBLE3_CONV2;  // 12 -> 12   3 x 12 = 36    x 273 = 9 828         36
typedef layer_size< 4, 4, true> UP2_CONV;       // 12 -> 12   3 x 3 = 9      x 1 057 = 9 513      144
typedef layer_size<24, 1, true> DOUBLE4_CONV1;  // 24 -> 8    1 x 8 = 8      x 1 057 = 8 456      216
typedef layer_size< 8, 1, true> DOUBLE4_CONV2;  // 8 -> 8     1 x 8 = 8      x 1 057 = 8 456       72
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 18, DSP_PACK_GROUP = 8;   // DSP48E2, 27-bit input: two int8 weights 18 bits apart, groups of 8

// The argument names become the kernel's register names: keep them.
void sp_unet_small_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                             hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                             hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
