#pragma once

#include "batchnorm_params_golden.hpp"
#include "nearest_neighbor_golden.hpp"
#include "conv2d_padded_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"

// A U-Net's up block, as trained:
//
//   nn.Sequential(Upsample(scale_factor=2, mode='nearest'),
//                 Conv2d(IN_CH, OUT_CH, 3, padding=1, bias=False), BatchNorm2d(OUT_CH), ReLU())
//
// H, W are the input's; the output is 2H x 2W. The conv is 3x3 in the trained
// SpecPW-UNets (up1.1.weight is [C][C][3][3]), though the architecture figure
// draws it as a 1x1. Members are in the state_dict's order (.1, .2).
template<typename data_t, int IN_CH, int OUT_CH>
struct up_block_params_golden {
    data_t w[OUT_CH][IN_CH][3][3];
    batchnorm_params_golden<data_t, OUT_CH> bn;
};

template<typename data_t, int H, int W, int IN_CH, int OUT_CH>
void up_block_golden(const data_t din[H][W][IN_CH],
                     const up_block_params_golden<data_t, IN_CH, OUT_CH> &p,
                     data_t eps,
                     data_t dout[2*H][2*W][OUT_CH]) {
    static const data_t no_bias[OUT_CH] = {};
    static data_t up[2*H][2*W][IN_CH];
    static data_t conv[2*H][2*W][OUT_CH];
    static data_t bn[2*H][2*W][OUT_CH];

    nearest_neighbor_golden<data_t, H, W, IN_CH, 2>(din, up);
    conv2d_padded_golden<data_t, 2*H, 2*W, IN_CH, OUT_CH, 3, 1>(up, p.w, no_bias, conv);
    batchnorm_golden<data_t, 2*H, 2*W, OUT_CH>(conv, p.bn.weight, p.bn.bias,
                                               p.bn.running_mean, p.bn.running_var, eps, bn);
    relu_golden<data_t, 2*H, 2*W, OUT_CH>(bn, dout);
}
