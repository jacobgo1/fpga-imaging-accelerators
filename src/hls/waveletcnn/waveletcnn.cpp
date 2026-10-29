#include "waveletcnn.hpp"

namespace waveletcnn_int8 { using int8_t = float; }
#include "../../../weights/quantized/waveletcnn/waveletcnn_int8.hpp"

namespace weights = waveletcnn_int8;

constexpr float LOWPASS_SCALE = frac_to_scale(weights::dwt_spectral_h0_frac);
constexpr float HIGHPASS_SCALE = frac_to_scale(weights::dwt_spectral_h1_frac);
constexpr float PROJECTION_WEIGHT_SCALE = frac_to_scale(weights::projection_weight_frac);
constexpr float PROJECTION_BIAS_SCALE = frac_to_scale(weights::projection_bias_frac);
constexpr float CONV_WEIGHT_SCALE = frac_to_scale(weights::conv_block_0_weight_frac);
constexpr float CONV_BIAS_SCALE = frac_to_scale(weights::conv_block_0_bias_frac);
constexpr float HEAD_WEIGHT_SCALE = frac_to_scale(weights::seg_head_0_weight_frac);
constexpr float HEAD_BIAS_SCALE = frac_to_scale(weights::seg_head_0_bias_frac);

void waveletcnn(data_t din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS],
                data_t dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES])
{
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Copy the patch on chip once, so the layers do not read DDR.
    static data_t patch[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS];
    for (int row = 0; row < GOLDEN_PATCH_SIZE; row++)
        for (int col = 0; col < GOLDEN_PATCH_SIZE; col++)
            for (int band = 0; band < GOLDEN_BANDS; band++)
                patch[row][col][band] = din[row][col][band];

    static data_t scores[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES];
    waveletcnn_golden<data_t, float, GOLDEN_PATCH_SIZE, GOLDEN_PATCH_SIZE, GOLDEN_BANDS,
                      GOLDEN_MID_CHANNELS, GOLDEN_CLASSES, GOLDEN_DWT_TAPS>(
        patch,
        weights::dwt_spectral_h0, LOWPASS_SCALE, weights::dwt_spectral_h1, HIGHPASS_SCALE,
        weights::projection_weight, PROJECTION_WEIGHT_SCALE, weights::projection_bias, PROJECTION_BIAS_SCALE,
        weights::conv_block_0_weight, CONV_WEIGHT_SCALE, weights::conv_block_0_bias, CONV_BIAS_SCALE,
        weights::seg_head_0_weight, HEAD_WEIGHT_SCALE, weights::seg_head_0_bias, HEAD_BIAS_SCALE,
        scores);

    for (int row = 0; row < GOLDEN_PATCH_SIZE; row++)
        for (int col = 0; col < GOLDEN_PATCH_SIZE; col++)
            for (int cls = 0; cls < GOLDEN_CLASSES; cls++)
                dout[row][col][cls] = scores[row][col][cls];
}
