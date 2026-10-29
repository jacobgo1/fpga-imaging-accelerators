#include "justoliunet.hpp"

namespace justoliunet_int8 { using int8_t = float; }
#include "../../../weights/quantized/justoliunet/justoliunet_int8.hpp"

namespace weights = justoliunet_int8;

constexpr float CONV1_WEIGHT_SCALE = frac_to_scale(weights::conv1_weight_frac);
constexpr float CONV2_WEIGHT_SCALE = frac_to_scale(weights::conv2_weight_frac);
constexpr float CONV3_WEIGHT_SCALE = frac_to_scale(weights::conv3_weight_frac);
constexpr float CONV4_WEIGHT_SCALE = frac_to_scale(weights::conv4_weight_frac);
constexpr float DENSE_WEIGHT_SCALE = frac_to_scale(weights::fc_weight_frac);
constexpr float CONV1_BIAS_SCALE = frac_to_scale(weights::conv1_bias_frac);
constexpr float CONV2_BIAS_SCALE = frac_to_scale(weights::conv2_bias_frac);
constexpr float CONV3_BIAS_SCALE = frac_to_scale(weights::conv3_bias_frac);
constexpr float CONV4_BIAS_SCALE = frac_to_scale(weights::conv4_bias_frac);
constexpr float DENSE_BIAS_SCALE = frac_to_scale(weights::fc_bias_frac);

void justoliunet(const data_t *din, data_t *dout, int n)
{
    // depth is for co-simulation only: the testbench's 32 pixels.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave depth=3520
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave depth=96
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=n      bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    pixels: for (int p = 0; p < n; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
        // Read the spectrum once, not once per multiply.
        data_t spectrum[JLN_BANDS][1];
        read: for (int band = 0; band < JLN_BANDS; band++)
            spectrum[band][0] = din[p * JLN_BANDS + band];

        data_t scores[JLN_CLASSES];
        justoliunet_golden<data_t, float, JLN_BANDS, JLN_BASE_CHANNELS, JLN_CLASSES, JLN_KERNEL_SIZE>(
            spectrum,
            weights::conv1_weight, CONV1_WEIGHT_SCALE, weights::conv1_bias, CONV1_BIAS_SCALE,
            weights::conv2_weight, CONV2_WEIGHT_SCALE, weights::conv2_bias, CONV2_BIAS_SCALE,
            weights::conv3_weight, CONV3_WEIGHT_SCALE, weights::conv3_bias, CONV3_BIAS_SCALE,
            weights::conv4_weight, CONV4_WEIGHT_SCALE, weights::conv4_bias, CONV4_BIAS_SCALE,
            weights::fc_weight, DENSE_WEIGHT_SCALE, weights::fc_bias, DENSE_BIAS_SCALE,
            scores);

        write: for (int c = 0; c < JLN_CLASSES; c++)
            dout[p * JLN_CLASSES + c] = scores[c];
    }
}
