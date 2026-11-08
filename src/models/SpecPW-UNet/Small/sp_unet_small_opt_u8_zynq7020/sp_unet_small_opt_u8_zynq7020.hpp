#ifndef SP_UNET_SMALL_OPT_U8_ZYNQ7020_HPP
#define SP_UNET_SMALL_OPT_U8_ZYNQ7020_HPP

#include "../sp_unet_small_opt_zynq7020/sp_unet_opt_zynq7000.hpp"
#include "../../specpw_unet_act8_fracs.hpp"

// SpecPW-UNet small on the Zynq-7020 with uint8 activations: every 3x3 conv two
// multiplies per DSP (conv3x3_tiled_ram, PACKED), the chain of sp_unet_small_opt_zynq7020
// otherwise. Bit for bit classify_fixed with act8_fracs('small') in tools/specpw_unet_model.py.
// The chain is in the .cpp.
constexpr int READ_PORTS = 1;   // 30 720 cycles a patch
// Every process takes at most T = 39 456 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 20;                         // convsp1: 20 bands a chunk, 6 chunks
constexpr int CONVSP1_PASSES = 6;                                   // ... x 6 passes over its pairs: 36 864 cycles a patch, 40 DSPs
constexpr int SKIP_DEPTH = 2048;                                    // the skip FIFO, pixels (room to spare)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 4 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 8, CONVSP2_OUT_PER_CYCLE = 1;  // convsp2 24 -> 12: 3 x 12 = 36 cycles a pixel (36 864), 8 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 4, CONVSP3_OUT_PER_CYCLE = 1;  // convsp3 12 -> 12: 3 x 12 = 36 (36 864), 4 uint8 multipliers in LUTs
constexpr bool CONVSP3_ON_DSPS = false;
constexpr int FINAL_IN_PER_CYCLE = 1, FINAL_OUT_PER_CYCLE = 1;      // final 8 -> 3: 8 x 3 = 24 (24 576), 1 in LUTs
constexpr bool FINAL_ON_DSPS = false;
// The 3x3 convs, packed: channels, taps x in x out steps a pixel x pixels = cycles a patch,
// DSPs (KT x PI x PO / 2). DSPs in all: 146 + 40 + 8 + 4 = 198.
typedef tiled_ram_layer_size<6, 2, 3, true> DOUBLE1_CONV1;          // 12 -> 12  3 x  2 x  6 =   36 a pixel  x 1058 =  38 088   18
typedef tiled_ram_layer_size<6, 2, 3, true> DOUBLE1_CONV2;          // 12 -> 12  3 x  2 x  6 =   36 a pixel  x 1058 =  38 088   18
typedef tiled_ram_layer_size<3, 2, 3, true> DOUBLE2_CONV1;          // 12 -> 24  3 x  4 x 12 =  144 a pixel  x  274 =  39 456    9
typedef tiled_ram_layer_size<6, 2, 3, true> DOUBLE2_CONV2;          // 24 -> 24  3 x  4 x 12 =  144 a pixel  x  274 =  39 456   18
typedef tiled_ram_layer_size<6, 2, 3, true> UP1_CONV;               // 24 -> 24  3 x  4 x 12 =  144 a pixel  x  274 =  39 456   18
typedef tiled_ram_layer_size<3, 2, 3, true> DOUBLE3_CONV1;          // 24 -> 12  3 x  8 x  6 =  144 a pixel  x  274 =  39 456    9
typedef tiled_ram_layer_size<2, 2, 3, true> DOUBLE3_CONV2;          // 12 -> 12  3 x  6 x  6 =  108 a pixel  x  274 =  29 592    6
typedef tiled_ram_layer_size<6, 2, 3, true> UP2_CONV;               // 12 -> 12  3 x  2 x  6 =   36 a pixel  x 1058 =  38 088   18
typedef tiled_ram_layer_size<8, 2, 3, true> DOUBLE4_CONV1;          // 24 ->  8  3 x  3 x  4 =   36 a pixel  x 1058 =  38 088   24
typedef tiled_ram_layer_size<8, 2, 1, true> DOUBLE4_CONV2;          //  8 ->  8  9 x  1 x  4 =   36 a pixel  x 1058 =  38 088    8
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 16;   // DSP48E1, 25-bit input: two weights 16 bits apart (dsp_pack.hpp)

namespace act8 = specpw_unet_act8_small;   // each layer's activations' fraction bits

// The argument names become the kernel's register names: keep them.
void sp_unet_small_opt_u8_zynq7020(const cube_word_t* din0, hls::burst_maxi<score_word_t> dout,
                                   int height, int width, uint32_t& cycles);

#endif
