#ifndef SP_UNET_MEDIUM_OPT_U8_ZYNQ7030_HPP
#define SP_UNET_MEDIUM_OPT_U8_ZYNQ7030_HPP

#include "../../Small/sp_unet_small_opt_zynq7020/sp_unet_opt_zynq7000.hpp"
#include "../../specpw_unet_act8_fracs.hpp"

// SpecPW-UNet medium on the Zynq-7030 (HYPSO-2) with uint8 activations: the chain of
// sp_unet_small_opt_u8_zynq7020, sized for 400 DSPs. Bit for bit classify_fixed with
// act8_fracs('medium') in tools/specpw_unet_model.py. The chain is in the .cpp.
constexpr int READ_PORTS = 1;   // 30 720 cycles a patch
// Every process takes at most T = 78 912 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 20;                         // convsp1: 20 bands a chunk, 6 chunks
constexpr int CONVSP1_PASSES = 12;                                  // ... x 12 passes over its pairs: 73 728 cycles a patch, 20 DSPs
constexpr int SKIP_DEPTH = 2048;                                    // the skip FIFO, pixels (room to spare)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 4 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 6, CONVSP2_OUT_PER_CYCLE = 1;  // convsp2 24 -> 16: 4 x 16 = 64 cycles a pixel (65 536), 6 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 4, CONVSP3_OUT_PER_CYCLE = 1;  // convsp3 16 -> 12: 4 x 12 = 48 (49 152), 4 DSPs
constexpr bool CONVSP3_ON_DSPS = true;
constexpr int FINAL_IN_PER_CYCLE = 1, FINAL_OUT_PER_CYCLE = 1;      // final 12 -> 3: 12 x 3 = 36 (36 864), 1 DSP
constexpr bool FINAL_ON_DSPS = true;
// The 3x3 convs, packed: channels, taps x in x out steps a pixel x pixels = cycles a patch,
// DSPs (KT x PI x PO / 2). DSPs in all: 252 + 20 + 11 + 4 = 287.
typedef tiled_ram_layer_size< 6, 2, 3, true> DOUBLE1_CONV1;         // 12 -> 24  3 x  2 x 12 =   72 a pixel  x 1058 =  76 176   18
typedef tiled_ram_layer_size<12, 2, 3, true> DOUBLE1_CONV2;         // 24 -> 24  3 x  2 x 12 =   72 a pixel  x 1058 =  76 176   36
typedef tiled_ram_layer_size< 6, 2, 3, true> DOUBLE2_CONV1;         // 24 -> 48  3 x  4 x 24 =  288 a pixel  x  274 =  78 912   18
typedef tiled_ram_layer_size<12, 2, 3, true> DOUBLE2_CONV2;         // 48 -> 48  3 x  4 x 24 =  288 a pixel  x  274 =  78 912   36
typedef tiled_ram_layer_size<12, 2, 3, true> UP1_CONV;              // 48 -> 48  3 x  4 x 24 =  288 a pixel  x  274 =  78 912   36
typedef tiled_ram_layer_size< 6, 2, 3, true> DOUBLE3_CONV1;         // 48 -> 24  3 x  8 x 12 =  288 a pixel  x  274 =  78 912   18
typedef tiled_ram_layer_size< 3, 2, 3, true> DOUBLE3_CONV2;         // 24 -> 24  3 x  8 x 12 =  288 a pixel  x  274 =  78 912    9
typedef tiled_ram_layer_size<12, 2, 3, true> UP2_CONV;              // 24 -> 24  3 x  2 x 12 =   72 a pixel  x 1058 =  76 176   36
typedef tiled_ram_layer_size<12, 2, 3, true> DOUBLE4_CONV1;         // 48 -> 12  3 x  4 x  6 =   72 a pixel  x 1058 =  76 176   36
typedef tiled_ram_layer_size< 3, 2, 3, true> DOUBLE4_CONV2;         // 12 -> 12  3 x  4 x  6 =   72 a pixel  x 1058 =  76 176    9
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 16;   // DSP48E1, 25-bit input: two weights 16 bits apart (dsp_pack.hpp)

namespace act8 = specpw_unet_act8_medium;   // each layer's activations' fraction bits

// The argument names become the kernel's register names: keep them.
void sp_unet_medium_opt_u8_zynq7030(const cube_word_t* din0, hls::burst_maxi<score_word_t> dout,
                                    int height, int width, uint32_t& cycles);

#endif
