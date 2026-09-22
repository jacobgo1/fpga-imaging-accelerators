#pragma once

#include "padding_golden.hpp"
#include "conv2d_golden.hpp"

// Convolution with zero padding: pad the input, then run the plain valid
// convolution on the padded buffer. With PAD = (K-1)/2 the output comes out
// the same size as the input ("same" convolution). Weight layout matches
// PyTorch's native Conv2d weight: [out][in][kh][kw].
template<typename data_t, int H, int W, int IN_CH, int OUT_CH, int K, int PAD>
void conv2d_padded_golden(data_t din[H][W][IN_CH],
                           data_t weight[OUT_CH][IN_CH][K][K],
                           data_t bias[OUT_CH],
                           data_t dout[H + 2*PAD - K + 1][W + 2*PAD - K + 1][OUT_CH]) {
    data_t padded[H + 2*PAD][W + 2*PAD][IN_CH];
    padding_golden<data_t, H, W, IN_CH, PAD>(din, padded);
    conv2d_golden<data_t, H + 2*PAD, W + 2*PAD, IN_CH, OUT_CH, K>(padded, weight, bias, dout);
}
