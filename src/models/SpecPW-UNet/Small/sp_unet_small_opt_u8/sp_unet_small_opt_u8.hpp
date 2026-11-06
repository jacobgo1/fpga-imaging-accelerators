#ifndef SP_UNET_SMALL_OPT_U8_HPP
#define SP_UNET_SMALL_OPT_U8_HPP

#include "../../Small/sp_unet_small_opt/sp_unet_opt.hpp"
#include "../../specpw_unet_act8_fracs.hpp"

// SpecPW-UNet small on the ZCU104 with uint8 activations: every 3x3 conv two multiplies per
// DSP (conv3x3_tiled, PACKED), the same chain as sp_unet_small_opt otherwise. Bit for bit
// classify_fixed with act8_fracs('small') in tools/specpw_unet_model.py. The chain is in the .cpp.
// Every process takes at most T = 6 552 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 32;    // convsp1: 32 bands a cycle, 4 cycles a pixel (4 096), 384 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs
constexpr int CONVSP2_OUT_PER_CYCLE = 2;        // convsp2 24 -> 12: 6 cycles a pixel (6 144), 48 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 2;        // convsp3 12 -> 12: 6 (6 144), 24 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 8 -> 3: 3 (3 072), 8 in LUTs
// The 3x3 convs, packed: channels, cycles a pixel x pixels (+ a row) = cycles a patch,
// DSPs (PO / 2 x PI x 9). DSPs in all: 891 + 384 + 72 = 1347.
typedef layer_size<12, 2, true> DOUBLE1_CONV1;  // 12 -> 12   1 x 6 = 6      x 1 057 = 6 342     108
typedef layer_size<12, 2, true> DOUBLE1_CONV2;  // 12 -> 12   1 x 6 = 6      x 1 057 = 6 342     108
typedef layer_size< 6, 2, true> DOUBLE2_CONV1;  // 12 -> 24   2 x 12 = 24    x 273 = 6 552        54
typedef layer_size<12, 2, true> DOUBLE2_CONV2;  // 24 -> 24   2 x 12 = 24    x 273 = 6 552       108
typedef layer_size<12, 2, true> UP1_CONV;       // 24 -> 24   2 x 12 = 24    x 273 = 6 552       108
typedef layer_size< 6, 2, true> DOUBLE3_CONV1;  // 24 -> 12   4 x 6 = 24     x 273 = 6 552        54
typedef layer_size< 3, 2, true> DOUBLE3_CONV2;  // 12 -> 12   4 x 6 = 24     x 273 = 6 552        27
typedef layer_size<12, 2, true> UP2_CONV;       // 12 -> 12   1 x 6 = 6      x 1 057 = 6 342     108
typedef layer_size< 8, 4, true> DOUBLE4_CONV1;  // 24 -> 8    3 x 2 = 6      x 1 057 = 6 342     144
typedef layer_size< 8, 2, true> DOUBLE4_CONV2;  // 8 -> 8     1 x 4 = 4      x 1 057 = 4 228      72
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 18;   // DSP48E2, 27-bit input: two weights 18 bits apart (dsp_pack.hpp)

namespace act8 = specpw_unet_act8_small;   // each layer's activations' fraction bits

// The argument names become the kernel's register names: keep them.
void sp_unet_small_opt_u8(const cube_word_t* din0, const cube_word_t* din1,
                       const cube_word_t* din2, const cube_word_t* din3,
                       hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
