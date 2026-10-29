#include "spectralpointwise1d.hpp"

namespace spectralpointwise1d_int8 { using int8_t = float; }
#include "../../../weights/quantized/spectralpointwise1d/spectralpointwise1d_int8.hpp"

namespace weights = spectralpointwise1d_int8;

constexpr float CONVSP1_WEIGHT_SCALE = frac_to_scale(weights::convsp1_weight_frac);
constexpr float CONVSP2_WEIGHT_SCALE = frac_to_scale(weights::convsp2_weight_frac);
constexpr float CONVSP3_WEIGHT_SCALE = frac_to_scale(weights::convsp3_weight_frac);
constexpr float CONVSP1_BIAS_SCALE = frac_to_scale(weights::convsp1_bias_frac);
constexpr float CONVSP2_BIAS_SCALE = frac_to_scale(weights::convsp2_bias_frac);
constexpr float CONVSP3_BIAS_SCALE = frac_to_scale(weights::convsp3_bias_frac);

void spectralpointwise1d(const data_t *din, data_t *dout, int n)
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
        data_t spectrum[SPW_BANDS];
        read: for (int band = 0; band < SPW_BANDS; band++)
            spectrum[band] = din[p * SPW_BANDS + band];

        data_t scores[SPW_CLASSES];
        spectralpointwise1d_golden<data_t, float, SPW_BANDS, SPW_CONV1_CHANNELS, SPW_CONV2_CHANNELS, SPW_CLASSES>(
            spectrum,
            weights::convsp1_weight, CONVSP1_WEIGHT_SCALE, weights::convsp1_bias, CONVSP1_BIAS_SCALE,
            weights::convsp2_weight, CONVSP2_WEIGHT_SCALE, weights::convsp2_bias, CONVSP2_BIAS_SCALE,
            weights::convsp3_weight, CONVSP3_WEIGHT_SCALE, weights::convsp3_bias, CONVSP3_BIAS_SCALE,
            scores);

        write: for (int c = 0; c < SPW_CLASSES; c++)
            dout[p * SPW_CLASSES + c] = scores[c];
    }
}
