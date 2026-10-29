#include "sp_unet_large.hpp"

namespace sp_unet_large_int8 { using int8_t = float; }
#include "../../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"

namespace weights = sp_unet_large_int8;

template<int FRAC> struct scale_of { static constexpr float value = frac_to_scale(FRAC); };

// A layer's weight, its scale, bias, its scale.
#define LAYER(name) weights::name##_weight, scale_of<weights::name##_weight_frac>::value, \
                    weights::name##_bias,   scale_of<weights::name##_bias_frac>::value

constexpr int PATCH_IN = GOLDEN_PATCH_SIZE * GOLDEN_PATCH_SIZE * GOLDEN_BANDS;
constexpr int PATCH_OUT = GOLDEN_PATCH_SIZE * GOLDEN_PATCH_SIZE * GOLDEN_CLASSES;

void sp_unet_large(const data_t *din, data_t *dout, int n)
{
    // depth is for co-simulation only: the testbench's 2 patches.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave depth=225280
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave depth=6144
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=n      bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    patches: for (int p = 0; p < n; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        // The stem reads the patch from DDR itself, each value once.
        static data_t scores[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES];
        specpw_unet_quant_golden<data_t, float, GOLDEN_PATCH_SIZE, GOLDEN_PATCH_SIZE, GOLDEN_BANDS, GOLDEN_CLASSES,
                                 model_size>(
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

        write: for (int row = 0; row < GOLDEN_PATCH_SIZE; row++)
            for (int col = 0; col < GOLDEN_PATCH_SIZE; col++)
                for (int c = 0; c < GOLDEN_CLASSES; c++)
                    dout[(long)p * PATCH_OUT + (row * GOLDEN_PATCH_SIZE + col) * GOLDEN_CLASSES + c] = scores[row][col][c];
    }
}

#undef LAYER
