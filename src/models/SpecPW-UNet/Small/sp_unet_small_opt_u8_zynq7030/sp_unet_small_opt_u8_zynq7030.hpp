#ifndef SP_UNET_SMALL_OPT_U8_ZYNQ7030_HPP
#define SP_UNET_SMALL_OPT_U8_ZYNQ7030_HPP

#include "../sp_unet_small_opt_zynq7020/sp_unet_opt_zynq7000.hpp"
#include "../../specpw_unet_act8_fracs.hpp"

// SpecPW-UNet small on the Zynq-7030 (HYPSO-2) with uint8 activations: the chain of
// sp_unet_small_opt_u8_zynq7020, sized for 400 DSPs, with two read ports. Bit for bit
// classify_fixed with act8_fracs('small') in tools/specpw_unet_model.py. The chain is in the .cpp.
constexpr int READ_PORTS = 2;   // 15 360 cycles a patch
// Every process takes at most T = 22 194 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 16;                         // convsp1: 16 bands a chunk, 7 chunks
constexpr int CONVSP1_PASSES = 3;                                   // ... x 3 passes over its pairs: 21 504 cycles a patch, 64 DSPs
constexpr int SKIP_DEPTH = 2048;                                    // the skip FIFO, pixels (room to spare)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 8 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 8, CONVSP2_OUT_PER_CYCLE = 2;  // convsp2 24 -> 12: 3 x 6 = 18 cycles a pixel (18 432), 16 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 4, CONVSP3_OUT_PER_CYCLE = 2;  // convsp3 12 -> 12: 3 x 6 = 18 (18 432), 8 uint8 multipliers in LUTs
constexpr bool CONVSP3_ON_DSPS = false;
constexpr int FINAL_IN_PER_CYCLE = 2, FINAL_OUT_PER_CYCLE = 1;      // final 8 -> 3: 4 x 3 = 12 (12 288), 2 in LUTs
constexpr bool FINAL_ON_DSPS = false;
// The 3x3 convs, packed: channels, taps x in x out steps a pixel x pixels = cycles a patch,
// DSPs (KT x PI x PO / 2). DSPs in all: 276 + 64 + 16 + 8 = 364.
typedef tiled_ram_layer_size<12, 2, 3, true> DOUBLE1_CONV1;         // 12 -> 12  3 x  1 x  6 =   18 a pixel  x 1058 =  19 044   36
typedef tiled_ram_layer_size<12, 2, 3, true> DOUBLE1_CONV2;         // 12 -> 12  3 x  1 x  6 =   18 a pixel  x 1058 =  19 044   36
typedef tiled_ram_layer_size< 4, 8, 1, true> DOUBLE2_CONV1;         // 12 -> 24  9 x  3 x  3 =   81 a pixel  x  274 =  22 194   16
typedef tiled_ram_layer_size< 8, 8, 1, true> DOUBLE2_CONV2;         // 24 -> 24  9 x  3 x  3 =   81 a pixel  x  274 =  22 194   32
typedef tiled_ram_layer_size< 8, 8, 1, true> UP1_CONV;              // 24 -> 24  9 x  3 x  3 =   81 a pixel  x  274 =  22 194   32
typedef tiled_ram_layer_size< 8, 4, 1, true> DOUBLE3_CONV1;         // 24 -> 12  9 x  3 x  3 =   81 a pixel  x  274 =  22 194   16
typedef tiled_ram_layer_size< 4, 4, 1, true> DOUBLE3_CONV2;         // 12 -> 12  9 x  3 x  3 =   81 a pixel  x  274 =  22 194    8
typedef tiled_ram_layer_size<12, 2, 3, true> UP2_CONV;              // 12 -> 12  3 x  1 x  6 =   18 a pixel  x 1058 =  19 044   36
typedef tiled_ram_layer_size< 8, 4, 3, true> DOUBLE4_CONV1;         // 24 ->  8  3 x  3 x  2 =   18 a pixel  x 1058 =  19 044   48
typedef tiled_ram_layer_size< 8, 4, 1, true> DOUBLE4_CONV2;         //  8 ->  8  9 x  1 x  2 =   18 a pixel  x 1058 =  19 044   16
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 16;   // DSP48E1, 25-bit input: two weights 16 bits apart (dsp_pack.hpp)

namespace act8 = specpw_unet_act8_small;   // each layer's activations' fraction bits

// The argument names become the kernel's register names: keep them.
void sp_unet_small_opt_u8_zynq7030(const cube_word_t* din0, const cube_word_t* din1,
                                   hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
