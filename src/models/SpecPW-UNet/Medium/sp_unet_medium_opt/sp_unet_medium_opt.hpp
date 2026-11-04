#ifndef SP_UNET_MEDIUM_OPT_DIRECT_HLS_HPP
#define SP_UNET_MEDIUM_OPT_DIRECT_HLS_HPP

#include "../../Small/sp_unet_small_opt/sp_unet_opt.hpp"

// SpecPW-UNet medium, direct convs, on the ZCU104: the chain and its sizes are in the .cpp.
// Every process takes at most T = 34 944 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;      // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 96 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs: in LUTs they took ~25 k LUTs (small)
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 24 -> 16: 16 cycles a pixel (16 384), 24 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 12: 12 (12 288), 16 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 12 -> 3: 3 (3 072), 12 in LUTs
// The convs: channels, cycles a pixel x pixels (+ a row) = cycles a patch, multipliers.
typedef layer_size< 3, 3, true> DOUBLE1_CONV1;  // 12 -> 24   4 x 8 = 32     x 1 057 = 33 824      81
typedef layer_size< 6, 3, true> DOUBLE1_CONV2;  // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef layer_size< 3, 3, true> DOUBLE2_CONV1;  // 24 -> 48   8 x 16 = 128   x 273 = 34 944        81
typedef layer_size< 6, 3, true> DOUBLE2_CONV2;  // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef layer_size< 6, 3, true> UP1_CONV;       // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef layer_size< 3, 3, true> DOUBLE3_CONV1;  // 48 -> 24   16 x 8 = 128   x 273 = 34 944        81
typedef layer_size< 6, 1, true> DOUBLE3_CONV2;  // 24 -> 24   4 x 24 = 96    x 273 = 26 208        54
typedef layer_size< 6, 3, true> UP2_CONV;       // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef layer_size< 6, 3, true> DOUBLE4_CONV1;  // 48 -> 12   8 x 4 = 32     x 1 057 = 33 824     162
typedef layer_size< 6, 1, true> DOUBLE4_CONV2;  // 12 -> 12   2 x 12 = 24    x 1 057 = 25 368      54
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 18;   // DSP48E2, 27-bit input: two weights 18 bits apart (dsp_pack.hpp)

// The argument names become the kernel's register names: keep them.
void sp_unet_medium_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                        hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                        hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
