#pragma once

#include "nearest_neighbor_golden.hpp"
#include "conv2d_same_quant_golden.hpp"

// up_block_golden.hpp on quantized weights, BatchNorm folded into the conv:
//
//   x = relu(conv3x3(nearest_upsample_x2(x)))
//
// H, W are the input's; the output is 2H x 2W. The upsampled map is UltraRAM
// (see conv_block_quant_golden.hpp).
template<typename act_t, typename w_t, int H, int W, int IN_CH, int OUT_CH>
void up_block_quant_golden(const act_t din[H][W][IN_CH],
                           const w_t weight[OUT_CH][IN_CH][3][3], act_t weight_scale,
                           const w_t bias[OUT_CH],                act_t bias_scale,
                           act_t dout[2*H][2*W][OUT_CH]) {
    static act_t up[2*H][2*W][IN_CH];
    #pragma HLS BIND_STORAGE variable=up type=ram_2p impl=uram
    nearest_neighbor_golden<act_t, H, W, IN_CH, 2>(din, up);
    conv2d_same_quant_golden<act_t, w_t, 2*H, 2*W, IN_CH, OUT_CH, 3, true>(
        up, weight, weight_scale, bias, bias_scale, dout);
}
