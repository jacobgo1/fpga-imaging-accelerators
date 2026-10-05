#include "justounetsimple.hpp"
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace q = justounetsimple_int8;

void justounetsimple(data_t din[JUNETS_H][JUNETS_W][JUNETS_IN_CH],
                     data_t dout[JUNETS_H][JUNETS_W][JUNETS_OUT_CH])
{
    // The patch in and the scores out go through one AXI master; the PS sets
    // their DDR addresses in the s_axilite registers before pulsing start.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Read the patch from DDR once. Otherwise every multiply in the
    // convolution would be a separate DDR access.
    static data_t patch[JUNETS_H][JUNETS_W][JUNETS_IN_CH];
    for (int i = 0; i < JUNETS_H; i++)
        for (int j = 0; j < JUNETS_W; j++)
            for (int c = 0; c < JUNETS_IN_CH; c++)
                patch[i][j][c] = din[i][j][c];

    static data_t scores[JUNETS_H][JUNETS_W][JUNETS_OUT_CH];
    justounetsimple_golden<data_t, int8_t, JUNETS_H, JUNETS_W, JUNETS_IN_CH,
                           JUNETS_BASE_CH, JUNETS_OUT_CH, JUNETS_K>(
        patch,
        q::conv1_weight, q::conv1_weight_frac, q::conv1_bias, q::conv1_bias_frac,
        q::conv2_weight, q::conv2_weight_frac, q::conv2_bias, q::conv2_bias_frac,
        q::conv3_weight, q::conv3_weight_frac, q::conv3_bias, q::conv3_bias_frac,
        q::conv4_weight, q::conv4_weight_frac, q::conv4_bias, q::conv4_bias_frac,
        scores);

    for (int i = 0; i < JUNETS_H; i++)
        for (int j = 0; j < JUNETS_W; j++)
            for (int c = 0; c < JUNETS_OUT_CH; c++)
                dout[i][j][c] = scores[i][j][c];
}
