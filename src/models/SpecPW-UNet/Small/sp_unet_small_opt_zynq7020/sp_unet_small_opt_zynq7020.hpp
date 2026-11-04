#ifndef SP_UNET_SMALL_OPT_DIRECT_ZYNQ7020_HLS_HPP
#define SP_UNET_SMALL_OPT_DIRECT_ZYNQ7020_HLS_HPP

#include "sp_unet_opt_zynq7020.hpp"

// SpecPW-UNet small, direct convs, on the Zynq-7020: the chain and its
// sizes are in the .cpp.
// Every process takes at most T = 76 176 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 20;                         // convsp1: 20 bands a chunk, 6 chunks
constexpr int CONVSP1_PASSES = 12;                                  // ... x 12 passes over its pairs: 73 728 cycles a patch, 20 DSPs
constexpr int SKIP_DEPTH = 2048;                                    // the skip FIFO, pixels (room to spare)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 4 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 4, CONVSP2_OUT_PER_CYCLE = 1;  // convsp2 24 -> 12: 6 x 12 = 72 cycles a pixel (73 728), 4 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 2, CONVSP3_OUT_PER_CYCLE = 1;  // convsp3 12 -> 12: 6 x 12 = 72 cycles a pixel (73 728), 2 DSPs
constexpr bool CONVSP3_ON_DSPS = true;
constexpr int FINAL_IN_PER_CYCLE = 1, FINAL_OUT_PER_CYCLE = 1;      // final 8 -> 3: 8 x 3 = 24 cycles a pixel (24 576), 1 DSP
constexpr bool FINAL_ON_DSPS = true;
// The convs: channels, taps x in x out steps a pixel x pixels = cycles a patch, multipliers.
typedef tiled_ram_layer_size< 6, 3, 1, true> DOUBLE1_CONV1;             // 12 -> 12  9 x  2 x  4 =   72 a pixel  x 1058 =  76 176   18
typedef tiled_ram_layer_size< 6, 3, 1, true> DOUBLE1_CONV2;             // 12 -> 12  9 x  2 x  4 =   72 a pixel  x 1058 =  76 176   18
typedef tiled_ram_layer_size<12, 1, 1, true> DOUBLE2_CONV1;             // 12 -> 24  9 x  1 x 24 =  216 a pixel  x  274 =  59 184   12
typedef tiled_ram_layer_size<24, 1, 1, true> DOUBLE2_CONV2;             // 24 -> 24  9 x  1 x 24 =  216 a pixel  x  274 =  59 184   24
typedef tiled_ram_layer_size<24, 1, 1, true> UP1_CONV;                  // 24 -> 24  9 x  1 x 24 =  216 a pixel  x  274 =  59 184   24
typedef tiled_ram_layer_size<12, 1, 1, true> DOUBLE3_CONV1;             // 24 -> 12  9 x  2 x 12 =  216 a pixel  x  274 =  59 184   12
typedef tiled_ram_layer_size< 6, 1, 1, true> DOUBLE3_CONV2;             // 12 -> 12  9 x  2 x 12 =  216 a pixel  x  274 =  59 184    6
typedef tiled_ram_layer_size< 6, 3, 1, true> UP2_CONV;                  // 12 -> 12  9 x  2 x  4 =   72 a pixel  x 1058 =  76 176   18
typedef tiled_ram_layer_size<24, 1, 1, true> DOUBLE4_CONV1;             // 24 ->  8  9 x  1 x  8 =   72 a pixel  x 1058 =  76 176   24
typedef tiled_ram_layer_size< 8, 1, 1, true> DOUBLE4_CONV2;             // 8 ->  8  9 x  1 x  8 =   72 a pixel  x 1058 =  76 176    8
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 16;   // DSP48E1, 25-bit input: two weights 16 bits apart (dsp_pack.hpp)

// The argument names become the kernel's register names: keep them.
void sp_unet_small_opt_zynq7020(const cube_word_t* din0, hls::burst_maxi<score_word_t> dout,
                                int height, int width, uint32_t& cycles);

#endif
