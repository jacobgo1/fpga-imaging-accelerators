#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// WaveletCNN's LeakyReLU after its 3x3 conv, fused with the 1x1 head (conv,
// folded BatchNorm, LeakyReLU): one pixel a cycle. Out: the scores, raster order.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MID_CHANNELS, int CLASSES, int CONV_SHIFT, int HEAD_BIAS_SHIFT,
         int HEAD_SHIFT, const wgt_t (&HEAD_WEIGHT)[CLASSES][MID_CHANNELS][1][1], const wgt_t (&HEAD_BIAS)[CLASSES],
         bool HEAD_DSP>
void leaky_head(hls::stream<vec_t<acc_t, MID_CHANNELS> >& in, hls::stream<vec_t<acc_t, CLASSES> >& out,
                int height, int width) {
    wgt_t weight[CLASSES][MID_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=0
    acc_t bias[CLASSES];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete
    load_w: for (int c = 0; c < CLASSES; c++) {
        #pragma HLS PIPELINE II=1
        bias[c] = (acc_t)HEAD_BIAS[c] * ((acc_t)1 << HEAD_BIAS_SHIFT);
        for (int ch = 0; ch < MID_CHANNELS; ch++) {
            #pragma HLS UNROLL
            weight[c][ch] = HEAD_WEIGHT[c][ch][0][0];
        }
    }

    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PATCH_HEIGHT * PATCH_WIDTH;
    pixel: for (int i = 0; i < pixels; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=680960
        const vec_t<acc_t, MID_CHANNELS> sums = in.read();
        act_t activations[MID_CHANNELS];
        #pragma HLS ARRAY_PARTITION variable=activations type=complete
        for (int ch = 0; ch < MID_CHANNELS; ch++) {
            #pragma HLS UNROLL
            activations[ch] = requant_leaky_relu<CONV_SHIFT>(sums.v[ch]);
        }
        vec_t<acc_t, CLASSES> scores;
        for (int c = 0; c < CLASSES; c++) {
            #pragma HLS UNROLL
            acc_t sum = bias[c];
            for (int ch = 0; ch < MID_CHANNELS; ch++) {
                #pragma HLS UNROLL
                sum += mul<HEAD_DSP, acc_t>(activations[ch], weight[c][ch]);
            }
            scores.v[c] = leaky_relu<>(round_shift<HEAD_SHIFT>(sum));
        }
        out.write(scores);
    }
}
