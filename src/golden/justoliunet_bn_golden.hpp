#pragma once

#include "conv1d_golden.hpp"
#include "batchnorm1d_golden.hpp"
#include "relu1d_golden.hpp"
#include "maxpool1d_golden.hpp"
#include "flatten_golden.hpp"
#include "dense_golden.hpp"

// 1D-Justo-LiuNet as trained, in float, with a BatchNorm after conv1: the
// testbenches' reference for justoliunet_golden.hpp. With an identity BatchNorm
// it is justoliunet, the model without one. Parameters in PyTorch's layouts.
template<typename data_t, int BANDS, int BASE, int CLASSES, int KERNEL>
void justoliunet_bn_golden(
    const data_t din[BANDS][1],
    const data_t conv1_weight[BASE][1][KERNEL],            const data_t conv1_bias[BASE],
    const data_t bn_weight[BASE],                          const data_t bn_bias[BASE],
    const data_t bn_mean[BASE],                            const data_t bn_var[BASE], data_t bn_eps,
    const data_t conv2_weight[2 * BASE][BASE][KERNEL],     const data_t conv2_bias[2 * BASE],
    const data_t conv3_weight[3 * BASE][2 * BASE][KERNEL], const data_t conv3_bias[3 * BASE],
    const data_t conv4_weight[4 * BASE][3 * BASE][KERNEL], const data_t conv4_bias[4 * BASE],
    const data_t dense_weight[CLASSES][4 * BASE * (((((BANDS - KERNEL + 1) / 2 - KERNEL + 1) / 2 - KERNEL + 1) / 2 - KERNEL + 1) / 2)],
    const data_t dense_bias[CLASSES],
    data_t dout[CLASSES])
{
    constexpr int CONV1 = BANDS - KERNEL + 1, POOL1 = CONV1 / 2;
    constexpr int CONV2 = POOL1 - KERNEL + 1, POOL2 = CONV2 / 2;
    constexpr int CONV3 = POOL2 - KERNEL + 1, POOL3 = CONV3 / 2;
    constexpr int CONV4 = POOL3 - KERNEL + 1, POOL4 = CONV4 / 2;
    constexpr int FLAT = 4 * BASE * POOL4;

    data_t conv1[CONV1][BASE], normalized1[CONV1][BASE], relu1[CONV1][BASE], pool1[POOL1][BASE];
    conv1d_golden<data_t, BANDS, 1, BASE, KERNEL>(din, conv1_weight, conv1_bias, conv1);
    batchnorm1d_golden<data_t, CONV1, BASE>(conv1, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, normalized1);
    relu1d_golden<data_t, CONV1, BASE>(normalized1, relu1);
    maxpool1d_golden<data_t, CONV1, BASE, 2>(relu1, pool1);

    data_t conv2[CONV2][2 * BASE], relu2[CONV2][2 * BASE], pool2[POOL2][2 * BASE];
    conv1d_golden<data_t, POOL1, BASE, 2 * BASE, KERNEL>(pool1, conv2_weight, conv2_bias, conv2);
    relu1d_golden<data_t, CONV2, 2 * BASE>(conv2, relu2);
    maxpool1d_golden<data_t, CONV2, 2 * BASE, 2>(relu2, pool2);

    data_t conv3[CONV3][3 * BASE], relu3[CONV3][3 * BASE], pool3[POOL3][3 * BASE];
    conv1d_golden<data_t, POOL2, 2 * BASE, 3 * BASE, KERNEL>(pool2, conv3_weight, conv3_bias, conv3);
    relu1d_golden<data_t, CONV3, 3 * BASE>(conv3, relu3);
    maxpool1d_golden<data_t, CONV3, 3 * BASE, 2>(relu3, pool3);

    data_t conv4[CONV4][4 * BASE], relu4[CONV4][4 * BASE], pool4[POOL4][4 * BASE];
    conv1d_golden<data_t, POOL3, 3 * BASE, 4 * BASE, KERNEL>(pool3, conv4_weight, conv4_bias, conv4);
    relu1d_golden<data_t, CONV4, 4 * BASE>(conv4, relu4);
    maxpool1d_golden<data_t, CONV4, 4 * BASE, 2>(relu4, pool4);

    data_t flat[FLAT];
    flatten_golden<data_t, POOL4, 4 * BASE>(pool4, flat);
    dense_golden<data_t, FLAT, CLASSES>(flat, dense_weight, dense_bias, dout);
}
