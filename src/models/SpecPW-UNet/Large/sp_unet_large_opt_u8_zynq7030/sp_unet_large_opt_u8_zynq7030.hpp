#ifndef SP_UNET_LARGE_OPT_U8_ZYNQ7030_HPP
#define SP_UNET_LARGE_OPT_U8_ZYNQ7030_HPP

#include "../../Small/sp_unet_small_opt_zynq7020/sp_unet_opt_zynq7000.hpp"
#include "../../specpw_unet_act8_fracs.hpp"

// SpecPW-UNet large on the Zynq-7030 (HYPSO-2) with uint8 activations: the chain of
// sp_unet_small_opt_u8_zynq7020, sized for 400 DSPs but kept to ~300, since the LUTs are
// what large is likely short of (the next faster table, 105 216 cycles a patch, needs 373).
// Bit for bit classify_fixed with act8_fracs('large') in tools/specpw_unet_model.py. The
// chain is in the .cpp.
constexpr int READ_PORTS = 1;   // 30 720 cycles a patch
// Every process takes at most T = 152 352 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 16;                         // convsp1: 16 bands a chunk, 7 chunks
constexpr int CONVSP1_PASSES = 16;                                  // ... x 16 passes over its pairs: 114 688 cycles a patch, 16 DSPs
constexpr int SKIP_DEPTH = 2048;                                    // the skip FIFO, pixels (room to spare)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 4 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 4, CONVSP2_OUT_PER_CYCLE = 1;  // convsp2 32 -> 16: 8 x 16 = 128 cycles a pixel (131 072), 4 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 2, CONVSP3_OUT_PER_CYCLE = 1;  // convsp3 16 -> 16: 8 x 16 = 128 (131 072), 2 DSPs
constexpr bool CONVSP3_ON_DSPS = true;
constexpr int FINAL_IN_PER_CYCLE = 1, FINAL_OUT_PER_CYCLE = 1;      // final 16 -> 3: 16 x 3 = 48 (49 152), 1 DSP
constexpr bool FINAL_ON_DSPS = true;
// The 3x3 convs, packed: channels, taps x in x out steps a pixel x pixels = cycles a patch,
// DSPs (KT x PI x PO / 2). DSPs in all: 276 + 16 + 7 + 4 = 303.
typedef tiled_ram_layer_size<16, 2, 1, true> DOUBLE1_CONV1;         // 16 -> 32  9 x  1 x 16 =  144 a pixel  x 1058 = 152 352   16
typedef tiled_ram_layer_size<32, 2, 1, true> DOUBLE1_CONV2;         // 32 -> 32  9 x  1 x 16 =  144 a pixel  x 1058 = 152 352   32
typedef tiled_ram_layer_size< 8, 2, 3, true> DOUBLE2_CONV1;         // 32 -> 64  3 x  4 x 32 =  384 a pixel  x  274 = 105 216   24
typedef tiled_ram_layer_size<16, 2, 3, true> DOUBLE2_CONV2;         // 64 -> 64  3 x  4 x 32 =  384 a pixel  x  274 = 105 216   48
typedef tiled_ram_layer_size<16, 2, 3, true> UP1_CONV;              // 64 -> 64  3 x  4 x 32 =  384 a pixel  x  274 = 105 216   48
typedef tiled_ram_layer_size< 8, 2, 3, true> DOUBLE3_CONV1;         // 64 -> 32  3 x  8 x 16 =  384 a pixel  x  274 = 105 216   24
typedef tiled_ram_layer_size< 4, 2, 3, true> DOUBLE3_CONV2;         // 32 -> 32  3 x  8 x 16 =  384 a pixel  x  274 = 105 216   12
typedef tiled_ram_layer_size<32, 2, 1, true> UP2_CONV;              // 32 -> 32  9 x  1 x 16 =  144 a pixel  x 1058 = 152 352   32
typedef tiled_ram_layer_size<32, 2, 1, true> DOUBLE4_CONV1;         // 64 -> 16  9 x  2 x  8 =  144 a pixel  x 1058 = 152 352   32
typedef tiled_ram_layer_size< 8, 2, 1, true> DOUBLE4_CONV2;         // 16 -> 16  9 x  2 x  8 =  144 a pixel  x 1058 = 152 352    8
}

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 16;   // DSP48E1, 25-bit input: two weights 16 bits apart (dsp_pack.hpp)

namespace act8 = specpw_unet_act8_large;   // each layer's activations' fraction bits

// The argument names become the kernel's register names: keep them.
void sp_unet_large_opt_u8_zynq7030(const cube_word_t* din0, hls::burst_maxi<score_word_t> dout,
                                   int height, int width, uint32_t& cycles);

#endif
