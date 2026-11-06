#ifndef SP_UNET_LARGE_OPT_U8_HPP
#define SP_UNET_LARGE_OPT_U8_HPP

#include "../../Small/sp_unet_small_opt/sp_unet_opt.hpp"
#include "../../specpw_unet_act8_fracs.hpp"

// SpecPW-UNet large on the ZCU104 with uint8 activations: every 3x3 conv two multiplies per
// DSP (conv3x3_tiled, PACKED), the same chain as sp_unet_large_opt otherwise. Bit for bit
// classify_fixed with act8_fracs('large') in tools/specpw_unet_model.py. The chain is in the .cpp.
// Every process takes at most T = 34 944 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;     // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 128 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 32 -> 16: 16 cycles a pixel (16 384), 32 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 16: 16 (16 384), 16 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 16 -> 3: 3 (3 072), 16 in LUTs
// The 3x3 convs, packed: channels, cycles a pixel x pixels (+ a row) = cycles a patch,
// DSPs (PO / 2 x PI x 9). DSPs in all: 1008 + 128 + 48 = 1184.
typedef layer_size< 8, 2, true> DOUBLE1_CONV1;  // 16 -> 32   2 x 16 = 32    x 1 057 = 33 824     72
typedef layer_size<16, 2, true> DOUBLE1_CONV2;  // 32 -> 32   2 x 16 = 32    x 1 057 = 33 824    144
typedef layer_size< 8, 2, true> DOUBLE2_CONV1;  // 32 -> 64   4 x 32 = 128   x 273 = 34 944       72
typedef layer_size<16, 2, true> DOUBLE2_CONV2;  // 64 -> 64   4 x 32 = 128   x 273 = 34 944      144
typedef layer_size<16, 2, true> UP1_CONV;       // 64 -> 64   4 x 32 = 128   x 273 = 34 944      144
typedef layer_size< 8, 2, true> DOUBLE3_CONV1;  // 64 -> 32   8 x 16 = 128   x 273 = 34 944       72
typedef layer_size< 4, 2, true> DOUBLE3_CONV2;  // 32 -> 32   8 x 16 = 128   x 273 = 34 944       36
typedef layer_size<16, 2, true> UP2_CONV;       // 32 -> 32   2 x 16 = 32    x 1 057 = 33 824    144
typedef layer_size<16, 2, true> DOUBLE4_CONV1;  // 64 -> 16   4 x 8 = 32     x 1 057 = 33 824    144
typedef layer_size< 4, 2, true> DOUBLE4_CONV2;  // 16 -> 16   4 x 8 = 32     x 1 057 = 33 824     36
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 18;   // DSP48E2, 27-bit input: two weights 18 bits apart (dsp_pack.hpp)

namespace act8 = specpw_unet_act8_large;   // each layer's activations' fraction bits

// The argument names become the kernel's register names: keep them.
void sp_unet_large_opt_u8(const cube_word_t* din0, const cube_word_t* din1,
                       const cube_word_t* din2, const cube_word_t* din3,
                       hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
