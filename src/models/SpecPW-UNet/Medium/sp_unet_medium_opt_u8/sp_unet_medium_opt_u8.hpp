#ifndef SP_UNET_MEDIUM_OPT_U8_HPP
#define SP_UNET_MEDIUM_OPT_U8_HPP

#include "../../Small/sp_unet_small_opt/sp_unet_opt.hpp"
#include "../../specpw_unet_act8_fracs.hpp"

// SpecPW-UNet medium on the ZCU104 with uint8 activations: every 3x3 conv two multiplies per
// DSP (conv3x3_tiled, PACKED), the same chain as sp_unet_medium_opt otherwise. Bit for bit
// classify_fixed with act8_fracs('medium') in tools/specpw_unet_model.py. The chain is in the .cpp.
// Every process takes at most T = 17 472 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;     // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 96 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 24 -> 16: 16 cycles a pixel (16 384), 24 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 12: 12 (12 288), 16 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 12 -> 3: 3 (3 072), 12 in LUTs
// The 3x3 convs, packed: channels, cycles a pixel x pixels (+ a row) = cycles a patch,
// DSPs (PO / 2 x PI x 9). DSPs in all: 1161 + 96 + 40 = 1297.
typedef layer_size< 3, 6, true> DOUBLE1_CONV1;  // 12 -> 24   4 x 4 = 16     x 1 057 = 16 912     81
typedef layer_size< 6, 6, true> DOUBLE1_CONV2;  // 24 -> 24   4 x 4 = 16     x 1 057 = 16 912    162
typedef layer_size< 3, 6, true> DOUBLE2_CONV1;  // 24 -> 48   8 x 8 = 64     x 273 = 17 472       81
typedef layer_size< 6, 6, true> DOUBLE2_CONV2;  // 48 -> 48   8 x 8 = 64     x 273 = 17 472      162
typedef layer_size< 6, 6, true> UP1_CONV;       // 48 -> 48   8 x 8 = 64     x 273 = 17 472      162
typedef layer_size< 3, 6, true> DOUBLE3_CONV1;  // 48 -> 24   16 x 4 = 64    x 273 = 17 472       81
typedef layer_size< 6, 2, true> DOUBLE3_CONV2;  // 24 -> 24   4 x 12 = 48    x 273 = 13 104       54
typedef layer_size< 6, 6, true> UP2_CONV;       // 24 -> 24   4 x 4 = 16     x 1 057 = 16 912    162
typedef layer_size< 6, 6, true> DOUBLE4_CONV1;  // 48 -> 12   8 x 2 = 16     x 1 057 = 16 912    162
typedef layer_size< 6, 2, true> DOUBLE4_CONV2;  // 12 -> 12   2 x 6 = 12     x 1 057 = 12 684     54
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 18;   // DSP48E2, 27-bit input: two weights 18 bits apart (dsp_pack.hpp)

namespace act8 = specpw_unet_act8_medium;   // each layer's activations' fraction bits

// The argument names become the kernel's register names: keep them.
void sp_unet_medium_opt_u8(const cube_word_t* din0, const cube_word_t* din1,
                       const cube_word_t* din2, const cube_word_t* din3,
                       hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
