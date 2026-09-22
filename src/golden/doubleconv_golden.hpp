#pragma once

#include "conv2d_padded_golden.hpp"
#include "relu_golden.hpp"

// Two consecutive (same-padded conv2d -> relu) passes, the classic
// "DoubleConv" block. Padding is fixed at (K-1)/2 so H and W never change
// between the input and the output. Weight layout matches PyTorch's native
// Conv2d weight: [out][in][kh][kw].
template<typename data_t, int H, int W, int IN_CH, int MID_CH, int OUT_CH, int K>
void doubleconv_golden(data_t din[H][W][IN_CH],
                        data_t weight1[MID_CH][IN_CH][K][K], data_t bias1[MID_CH],
                        data_t weight2[OUT_CH][MID_CH][K][K], data_t bias2[OUT_CH],
                        data_t dout[H][W][OUT_CH]) {
    constexpr int PAD = (K - 1) / 2;

    data_t conv1_out[H][W][MID_CH];
    data_t relu1_out[H][W][MID_CH];
    data_t conv2_out[H][W][OUT_CH];

    conv2d_padded_golden<data_t, H, W, IN_CH, MID_CH, K, PAD>(din, weight1, bias1, conv1_out);
    relu_golden<data_t, H, W, MID_CH>(conv1_out, relu1_out);
    conv2d_padded_golden<data_t, H, W, MID_CH, OUT_CH, K, PAD>(relu1_out, weight2, bias2, conv2_out);
    relu_golden<data_t, H, W, OUT_CH>(conv2_out, dout);
}
