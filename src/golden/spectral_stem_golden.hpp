#pragma once

#include "batchnorm_params_golden.hpp"
#include "conv2d_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"

// SpecPW-UNet's spectral stem: three 1x1 convs, so each pixel's spectrum is
// mixed on its own, without looking at its neighbours:
//
//   x = relu(bnsp1(convsp1(x)))      IN_CH -> S1
//   x = relu(convsp2(x))             S1 -> S2
//   x = relu(convsp3(x))             S2 -> S3
//
// Members are in the state_dict's order (convsp1, bnsp1, convsp2, convsp3).
template<typename data_t, int IN_CH, int S1, int S2, int S3>
struct spectral_stem_params_golden {
    data_t w1[S1][IN_CH][1][1];
    data_t b1[S1];
    batchnorm_params_golden<data_t, S1> bn1;
    data_t w2[S2][S1][1][1];
    data_t b2[S2];
    data_t w3[S3][S2][1][1];
    data_t b3[S3];
};

template<typename data_t, int H, int W, int IN_CH, int S1, int S2, int S3>
void spectral_stem_golden(const data_t din[H][W][IN_CH],
                          const spectral_stem_params_golden<data_t, IN_CH, S1, S2, S3> &p,
                          data_t eps,
                          data_t dout[H][W][S3]) {
    static data_t conv1[H][W][S1];
    static data_t bn1[H][W][S1];
    static data_t act1[H][W][S1];
    static data_t conv2[H][W][S2];
    static data_t act2[H][W][S2];
    static data_t conv3[H][W][S3];

    // a 1x1 valid conv is the same size as its input
    conv2d_golden<data_t, H, W, IN_CH, S1, 1>(din, p.w1, p.b1, conv1);
    batchnorm_golden<data_t, H, W, S1>(conv1, p.bn1.weight, p.bn1.bias,
                                       p.bn1.running_mean, p.bn1.running_var, eps, bn1);
    relu_golden<data_t, H, W, S1>(bn1, act1);

    conv2d_golden<data_t, H, W, S1, S2, 1>(act1, p.w2, p.b2, conv2);
    relu_golden<data_t, H, W, S2>(conv2, act2);

    conv2d_golden<data_t, H, W, S2, S3, 1>(act2, p.w3, p.b3, conv3);
    relu_golden<data_t, H, W, S3>(conv3, dout);
}
