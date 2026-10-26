#include "sp_unet_small.hpp"
#include <type_traits>

// The quantized weights, read with the header's int8_t meaning float: the
// header declares its arrays inside this namespace, so its unqualified int8_t
// finds this alias first. The ROM then holds the same integers as exact floats,
// and the hardware multiplies float by float, with no int-to-float converter
// (as in justounetsimple.cpp: Vitis 2025.2 leaves that converter's module out
// of the exported IP).
namespace sp_unet_small_int8 { using int8_t = float; }
#include "../../../weights/quantized/sp_unet_small/sp_unet_small_int8.hpp"

namespace q = sp_unet_small_int8;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::convsp1_weight)>::type>::type, float>::value,
              "the weights must be read as float; did the header's element type change?");

// A tensor's 2^-frac, as a compile-time constant.
template<int FRAC> struct scale_of { static constexpr float value = pow2_neg(FRAC); };

// One layer's arguments: weight, its scale, bias, its scale.
#define LAYER(name) q::name##_weight, scale_of<q::name##_weight_frac>::value, \
                    q::name##_bias,   scale_of<q::name##_bias_frac>::value

constexpr int PATCH_IN = SPU_H * SPU_W * SPU_IN_CH, PATCH_OUT = SPU_H * SPU_W * SPU_OUT_CH;

void sp_unet_small(const data_t *din, data_t *dout, int n)
{
    // The patches in and the scores out go through one AXI master; the PS sets
    // their DDR addresses and n in the s_axilite registers before pulsing start.
    // depth is only for co-simulation: the testbench's 2 patches.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave depth=225280
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave depth=6144
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=n      bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    patches: for (int p = 0; p < n; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        // The stem reads the patch from DDR itself, each value once.
        static data_t scores[SPU_H][SPU_W][SPU_OUT_CH];
        specpw_unet_quant_golden<data_t, float, SPU_H, SPU_W, SPU_IN_CH, SPU_OUT_CH, spu_size>(
            din, (long)p * PATCH_IN,
            LAYER(convsp1), LAYER(convsp2), LAYER(convsp3),
            LAYER(double1_block_0), LAYER(double1_block_3),
            LAYER(double2_block_0), LAYER(double2_block_3),
            LAYER(up1_1),
            LAYER(double3_block_0), LAYER(double3_block_3),
            LAYER(up2_1),
            LAYER(double4_block_0), LAYER(double4_block_3),
            LAYER(final),
            scores);

        write: for (int i = 0; i < SPU_H; i++)
            for (int j = 0; j < SPU_W; j++)
                for (int c = 0; c < SPU_OUT_CH; c++)
                    dout[(long)p * PATCH_OUT + (i * SPU_W + j) * SPU_OUT_CH + c] = scores[i][j][c];
    }
}

#undef LAYER
