#pragma once

#include "conv2d_same_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"
#include "maxpool_golden.hpp"
#include "upsample_golden.hpp"

// 2D-JustoUNet as trained, in float, with its BatchNorms: the testbench's
// reference for justounetsimple_golden.hpp. Four 3x3 conv blocks:
//
//   A: conv(BANDS -> BASE)      BN, ReLU, maxpool                 H/2
//   B: conv(BASE -> 2 BASE)     BN, ReLU, maxpool, upsample       H/2
//   C: conv(2 BASE -> BASE)     BN, ReLU, upsample                H
//   D: conv(BASE -> CLASSES)    BN: the scores
//
// Parameters in PyTorch's layouts. HEIGHT and WIDTH must be multiples of 4.
template<typename data_t, int HEIGHT, int WIDTH, int BANDS, int BASE, int CLASSES, int KERNEL>
void justounet2d_golden(
    data_t din[HEIGHT][WIDTH][BANDS],
    data_t conv1_weight[BASE][BANDS][KERNEL][KERNEL], data_t conv1_bias[BASE],
    data_t bn1_weight[BASE], data_t bn1_bias[BASE], data_t bn1_mean[BASE], data_t bn1_var[BASE],
    data_t conv2_weight[2 * BASE][BASE][KERNEL][KERNEL], data_t conv2_bias[2 * BASE],
    data_t bn2_weight[2 * BASE], data_t bn2_bias[2 * BASE], data_t bn2_mean[2 * BASE], data_t bn2_var[2 * BASE],
    data_t conv3_weight[BASE][2 * BASE][KERNEL][KERNEL], data_t conv3_bias[BASE],
    data_t bn3_weight[BASE], data_t bn3_bias[BASE], data_t bn3_mean[BASE], data_t bn3_var[BASE],
    data_t conv4_weight[CLASSES][BASE][KERNEL][KERNEL], data_t conv4_bias[CLASSES],
    data_t bn4_weight[CLASSES], data_t bn4_bias[CLASSES], data_t bn4_mean[CLASSES], data_t bn4_var[CLASSES],
    data_t eps,
    data_t dout[HEIGHT][WIDTH][CLASSES])
{
    constexpr int HALF_H = HEIGHT / 2, HALF_W = WIDTH / 2;
    constexpr int QUARTER_H = HEIGHT / 4, QUARTER_W = WIDTH / 4;

    static data_t conv_a[HEIGHT][WIDTH][BASE], normalized_a[HEIGHT][WIDTH][BASE], relu_a[HEIGHT][WIDTH][BASE];
    static data_t pooled_a[HALF_H][HALF_W][BASE];
    conv2d_same_golden<data_t, HEIGHT, WIDTH, BANDS, BASE, KERNEL>(din, conv1_weight, conv1_bias, conv_a);
    batchnorm_golden<data_t, HEIGHT, WIDTH, BASE>(conv_a, bn1_weight, bn1_bias, bn1_mean, bn1_var, eps, normalized_a);
    relu_golden<data_t, HEIGHT, WIDTH, BASE>(normalized_a, relu_a);
    maxpool_golden<data_t, HEIGHT, WIDTH, BASE>(relu_a, pooled_a);

    static data_t conv_b[HALF_H][HALF_W][2 * BASE], normalized_b[HALF_H][HALF_W][2 * BASE];
    static data_t relu_b[HALF_H][HALF_W][2 * BASE], pooled_b[QUARTER_H][QUARTER_W][2 * BASE];
    static data_t upsampled_b[HALF_H][HALF_W][2 * BASE];
    conv2d_same_golden<data_t, HALF_H, HALF_W, BASE, 2 * BASE, KERNEL>(pooled_a, conv2_weight, conv2_bias, conv_b);
    batchnorm_golden<data_t, HALF_H, HALF_W, 2 * BASE>(conv_b, bn2_weight, bn2_bias, bn2_mean, bn2_var, eps, normalized_b);
    relu_golden<data_t, HALF_H, HALF_W, 2 * BASE>(normalized_b, relu_b);
    maxpool_golden<data_t, HALF_H, HALF_W, 2 * BASE>(relu_b, pooled_b);
    upsample_golden<data_t, QUARTER_H, QUARTER_W, 2 * BASE>(pooled_b, upsampled_b);

    static data_t conv_c[HALF_H][HALF_W][BASE], normalized_c[HALF_H][HALF_W][BASE], relu_c[HALF_H][HALF_W][BASE];
    static data_t upsampled_c[HEIGHT][WIDTH][BASE];
    conv2d_same_golden<data_t, HALF_H, HALF_W, 2 * BASE, BASE, KERNEL>(upsampled_b, conv3_weight, conv3_bias, conv_c);
    batchnorm_golden<data_t, HALF_H, HALF_W, BASE>(conv_c, bn3_weight, bn3_bias, bn3_mean, bn3_var, eps, normalized_c);
    relu_golden<data_t, HALF_H, HALF_W, BASE>(normalized_c, relu_c);
    upsample_golden<data_t, HALF_H, HALF_W, BASE>(relu_c, upsampled_c);

    static data_t conv_d[HEIGHT][WIDTH][CLASSES];
    conv2d_same_golden<data_t, HEIGHT, WIDTH, BASE, CLASSES, KERNEL>(upsampled_c, conv4_weight, conv4_bias, conv_d);
    batchnorm_golden<data_t, HEIGHT, WIDTH, CLASSES>(conv_d, bn4_weight, bn4_bias, bn4_mean, bn4_var, eps, dout);
}
