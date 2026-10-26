#pragma once

#include "conv2d_same_quant_golden.hpp"

// conv_block_golden.hpp on quantized weights, BatchNorm folded into each conv
// (tools/quantize_weights.py), so each conv has a bias and BN+ReLU is a ReLU:
//
//   x = relu(conv3x3(x));  x = relu(conv3x3(x))
//
// The map between the two convs is UltraRAM: in the large SpecPW-UNet it is
// 32 x 32 x 32 floats, and BRAM is left for the weights (ROM cannot be URAM).
template<typename act_t, typename w_t, int H, int W, int IN_CH, int OUT_CH>
void conv_block_quant_golden(const act_t din[H][W][IN_CH],
                             const w_t w1[OUT_CH][IN_CH][3][3],  act_t w1_scale,
                             const w_t b1[OUT_CH],               act_t b1_scale,
                             const w_t w2[OUT_CH][OUT_CH][3][3], act_t w2_scale,
                             const w_t b2[OUT_CH],               act_t b2_scale,
                             act_t dout[H][W][OUT_CH]) {
    static act_t mid[H][W][OUT_CH];
    #pragma HLS BIND_STORAGE variable=mid type=ram_2p impl=uram
    conv2d_same_quant_golden<act_t, w_t, H, W, IN_CH, OUT_CH, 3, true>(din, w1, w1_scale, b1, b1_scale, mid);
    conv2d_same_quant_golden<act_t, w_t, H, W, OUT_CH, OUT_CH, 3, true>(mid, w2, w2_scale, b2, b2_scale, dout);
}
