#ifndef SP_UNET_LARGE_OPT_DIRECT_HLS_HPP
#define SP_UNET_LARGE_OPT_DIRECT_HLS_HPP

#include "../../Small/sp_unet_small_opt/sp_unet_opt.hpp"

// SpecPW-UNet large, direct convs, on the ZCU104: the chain and its sizes are in the .cpp.
// Every process takes at most T = 69 888 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;      // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 128 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs: in LUTs they took ~25 k LUTs (small)
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 32 -> 16: 16 cycles a pixel (16 384), 32 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 16: 16 (16 384), 16 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 16 -> 3: 3 (3 072), 16 DSPs
// The convs: channels, cycles a pixel x pixels (+ a row) = cycles a patch, multipliers.
typedef layer_size< 8, 1, true> DOUBLE1_CONV1;  // 16 -> 32   2 x 32 = 64    x 1 057 = 67 648      72
typedef layer_size<16, 1, true> DOUBLE1_CONV2;  // 32 -> 32   2 x 32 = 64    x 1 057 = 67 648     144
typedef layer_size< 8, 1, true> DOUBLE2_CONV1;  // 32 -> 64   4 x 64 = 256   x 273 = 69 888        72
typedef layer_size<16, 1, true> DOUBLE2_CONV2;  // 64 -> 64   4 x 64 = 256   x 273 = 69 888       144
typedef layer_size<16, 1, true> UP1_CONV;       // 64 -> 64   4 x 64 = 256   x 273 = 69 888       144
typedef layer_size< 8, 1, true> DOUBLE3_CONV1;  // 64 -> 32   8 x 32 = 256   x 273 = 69 888        72
typedef layer_size< 4, 1, true> DOUBLE3_CONV2;  // 32 -> 32   8 x 32 = 256   x 273 = 69 888        36
typedef layer_size<16, 1, true> UP2_CONV;       // 32 -> 32   2 x 32 = 64    x 1 057 = 67 648     144
typedef layer_size<16, 1, true> DOUBLE4_CONV1;  // 64 -> 16   4 x 16 = 64    x 1 057 = 67 648     144
typedef layer_size< 4, 1, true> DOUBLE4_CONV2;  // 16 -> 16   4 x 16 = 64    x 1 057 = 67 648      36
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 18;   // DSP48E2, 27-bit input: two weights 18 bits apart (dsp_pack.hpp)

// The argument names become the kernel's register names: keep them.
void sp_unet_large_opt(const cube_word_t* din0, const cube_word_t* din1,
                       const cube_word_t* din2, const cube_word_t* din3,
                       hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
