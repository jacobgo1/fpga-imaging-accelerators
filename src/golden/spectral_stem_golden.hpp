#pragma once

#include "batchnorm_params_golden.hpp"
#include "conv2d_same_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"

// SpecPW-UNet's spectral stem, three 1x1 convs:
//   relu(bnsp1(convsp1(x))), relu(convsp2(x)), relu(convsp3(x))
// Members in the state_dict's order.
template<typename data_t, int IN_CHANNELS, int STEM1, int STEM2, int STEM3>
struct spectral_stem_params_golden {
    data_t conv1_weight[STEM1][IN_CHANNELS][1][1];
    data_t conv1_bias[STEM1];
    batchnorm_params_golden<data_t, STEM1> bn1;
    data_t conv2_weight[STEM2][STEM1][1][1];
    data_t conv2_bias[STEM2];
    data_t conv3_weight[STEM3][STEM2][1][1];
    data_t conv3_bias[STEM3];
};

template<typename data_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int STEM1, int STEM2, int STEM3>
void spectral_stem_golden(const data_t din[HEIGHT][WIDTH][IN_CHANNELS],
                          const spectral_stem_params_golden<data_t, IN_CHANNELS, STEM1, STEM2, STEM3> &params,
                          data_t eps,
                          data_t dout[HEIGHT][WIDTH][STEM3]) {
    static data_t conv1[HEIGHT][WIDTH][STEM1], normalized1[HEIGHT][WIDTH][STEM1], relu1[HEIGHT][WIDTH][STEM1];
    static data_t conv2[HEIGHT][WIDTH][STEM2], relu2[HEIGHT][WIDTH][STEM2];
    static data_t conv3[HEIGHT][WIDTH][STEM3];
    const auto &bn1 = params.bn1;

    conv2d_same_golden<data_t, HEIGHT, WIDTH, IN_CHANNELS, STEM1, 1>(din, params.conv1_weight, params.conv1_bias, conv1);
    batchnorm_golden<data_t, HEIGHT, WIDTH, STEM1>(
        conv1, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var, eps, normalized1);
    relu_golden<data_t, HEIGHT, WIDTH, STEM1>(normalized1, relu1);

    conv2d_same_golden<data_t, HEIGHT, WIDTH, STEM1, STEM2, 1>(relu1, params.conv2_weight, params.conv2_bias, conv2);
    relu_golden<data_t, HEIGHT, WIDTH, STEM2>(conv2, relu2);

    conv2d_same_golden<data_t, HEIGHT, WIDTH, STEM2, STEM3, 1>(relu2, params.conv3_weight, params.conv3_bias, conv3);
    relu_golden<data_t, HEIGHT, WIDTH, STEM3>(conv3, dout);
}
