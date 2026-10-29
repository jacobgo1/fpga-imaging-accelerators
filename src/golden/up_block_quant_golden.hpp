#pragma once

#include "upsample_golden.hpp"
#include "conv2d_same_quant_golden.hpp"

// up_block_golden.hpp with BatchNorm folded into the conv. HEIGHT and WIDTH are the input's.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void up_block_quant_golden(const act_t din[HEIGHT][WIDTH][IN_CHANNELS],
                           const weight_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], act_t weight_scale,
                           const weight_t bias[OUT_CHANNELS],                      act_t bias_scale,
                           act_t dout[2 * HEIGHT][2 * WIDTH][OUT_CHANNELS]) {
    static act_t upsampled[2 * HEIGHT][2 * WIDTH][IN_CHANNELS];
    #pragma HLS BIND_STORAGE variable=upsampled type=ram_2p impl=uram
    upsample_golden<act_t, HEIGHT, WIDTH, IN_CHANNELS>(din, upsampled);
    conv2d_same_quant_golden<act_t, weight_t, 2 * HEIGHT, 2 * WIDTH, IN_CHANNELS, OUT_CHANNELS, 3, true>(
        upsampled, weight, weight_scale, bias, bias_scale, dout);
}
