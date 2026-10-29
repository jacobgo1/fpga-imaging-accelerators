#include "justounetsimple.hpp"

namespace justounetsimple_int8 { using int8_t = float; }
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace weights = justounetsimple_int8;

constexpr float CONV1_WEIGHT_SCALE = frac_to_scale(weights::conv1_weight_frac);
constexpr float CONV2_WEIGHT_SCALE = frac_to_scale(weights::conv2_weight_frac);
constexpr float CONV3_WEIGHT_SCALE = frac_to_scale(weights::conv3_weight_frac);
constexpr float CONV4_WEIGHT_SCALE = frac_to_scale(weights::conv4_weight_frac);
constexpr float CONV1_BIAS_SCALE = frac_to_scale(weights::conv1_bias_frac);
constexpr float CONV2_BIAS_SCALE = frac_to_scale(weights::conv2_bias_frac);
constexpr float CONV3_BIAS_SCALE = frac_to_scale(weights::conv3_bias_frac);
constexpr float CONV4_BIAS_SCALE = frac_to_scale(weights::conv4_bias_frac);

void justounetsimple(data_t din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS],
                     data_t dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES])
{
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Copy the patch on chip once, so the convolution does not read DDR.
    static data_t patch[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS];
    for (int row = 0; row < GOLDEN_PATCH_SIZE; row++)
        for (int col = 0; col < GOLDEN_PATCH_SIZE; col++)
            for (int band = 0; band < GOLDEN_BANDS; band++)
                patch[row][col][band] = din[row][col][band];

    static data_t scores[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES];
    justounetsimple_golden<data_t, float, GOLDEN_PATCH_SIZE, GOLDEN_PATCH_SIZE, GOLDEN_BANDS,
                           GOLDEN_BASE_CHANNELS, GOLDEN_CLASSES, GOLDEN_KERNEL_SIZE>(
        patch,
        weights::conv1_weight, CONV1_WEIGHT_SCALE, weights::conv1_bias, CONV1_BIAS_SCALE,
        weights::conv2_weight, CONV2_WEIGHT_SCALE, weights::conv2_bias, CONV2_BIAS_SCALE,
        weights::conv3_weight, CONV3_WEIGHT_SCALE, weights::conv3_bias, CONV3_BIAS_SCALE,
        weights::conv4_weight, CONV4_WEIGHT_SCALE, weights::conv4_bias, CONV4_BIAS_SCALE,
        scores);

    for (int row = 0; row < GOLDEN_PATCH_SIZE; row++)
        for (int col = 0; col < GOLDEN_PATCH_SIZE; col++)
            for (int cls = 0; cls < GOLDEN_CLASSES; cls++)
                dout[row][col][cls] = scores[row][col][cls];
}
