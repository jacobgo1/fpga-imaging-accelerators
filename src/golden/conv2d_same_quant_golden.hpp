#pragma once

// Weights are the quantized integers q with one power-of-two scale per tensor:
// real value = q * 2^-frac (weights/quantized/). The scale is applied once per
// output value, after the sum, so the inner loop multiplies by q itself:
//
//   out = (sum of act * q) * weight_scale + bias_q * bias_scale,   scale = 2^-frac
//
// w_t can be int8_t, or float holding the same integers (what the HLS kernels
// use). Layout: weights [out][in][kh][kw], activations HWC.

// Same-padded KxK convolution (zero padding) on quantized weights. Out-of-range
// taps are skipped instead of copying the input into a padded buffer, so
// nothing larger than the input and output maps is needed. RELU applies a
// ReLU to each output as it is written, so it needs no buffer of its own.
template<typename act_t, typename w_t, int H, int W, int IN_CH, int OUT_CH, int K, bool RELU = false>
void conv2d_same_quant_golden(const act_t din[H][W][IN_CH],
                              const w_t weight[OUT_CH][IN_CH][K][K], act_t weight_scale,
                              const w_t bias[OUT_CH], act_t bias_scale,
                              act_t dout[H][W][OUT_CH]) {
    constexpr int PAD = (K - 1) / 2;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int oc = 0; oc < OUT_CH; oc++) {
                act_t sum = 0;
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            int y = i + ki - PAD;
                            int x = j + kj - PAD;
                            if (y < 0 || y >= H || x < 0 || x >= W) continue;
                            sum += din[y][x][ic] * (act_t)weight[oc][ic][ki][kj];
                        }
                    }
                }
                act_t v = sum * weight_scale + (act_t)bias[oc] * bias_scale;
                dout[i][j][oc] = (RELU && v < 0) ? (act_t)0 : v;
            }
        }
    }
}
