#pragma once

#include "../../../golden/layers.hpp"

using data_t = float;

constexpr int GOLDEN_PATCH_SIZE = 32;
constexpr int GOLDEN_BANDS = 110;
constexpr int GOLDEN_BASE_CHANNELS = 6;
constexpr int GOLDEN_CLASSES = 3;
constexpr int GOLDEN_KERNEL_SIZE = 3;


void justounetsimple(data_t din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS],
                     data_t dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES]);

// The quantized 2D-JustoUNet: tb/justounetsimple/justounet2d_golden.hpp with the BatchNorms folded
// into the convs and int8 weights. HEIGHT and WIDTH must be multiples of 4.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int BANDS, int BASE, int CLASSES, int KERNEL>
void justounetsimple_golden(
    const act_t din[HEIGHT][WIDTH][BANDS],
    const weight_t conv1_weight[BASE][BANDS][KERNEL][KERNEL],     act_t conv1_weight_scale,
    const weight_t conv1_bias[BASE],                              act_t conv1_bias_scale,
    const weight_t conv2_weight[2 * BASE][BASE][KERNEL][KERNEL], act_t conv2_weight_scale,
    const weight_t conv2_bias[2 * BASE],                         act_t conv2_bias_scale,
    const weight_t conv3_weight[BASE][2 * BASE][KERNEL][KERNEL], act_t conv3_weight_scale,
    const weight_t conv3_bias[BASE],                             act_t conv3_bias_scale,
    const weight_t conv4_weight[CLASSES][BASE][KERNEL][KERNEL],  act_t conv4_weight_scale,
    const weight_t conv4_bias[CLASSES],                          act_t conv4_bias_scale,
    act_t dout[HEIGHT][WIDTH][CLASSES])
{
    constexpr int HALF_H = HEIGHT / 2, HALF_W = WIDTH / 2;
    constexpr int QUARTER_H = HEIGHT / 4, QUARTER_W = WIDTH / 4;
    using golden::activation;

    static act_t relu_a[HEIGHT][WIDTH][BASE], pooled_a[HALF_H][HALF_W][BASE];
    golden::conv2d<HEIGHT, WIDTH, BANDS, BASE, KERNEL, activation::relu>(
        din, conv1_weight, conv1_bias, relu_a, conv1_weight_scale, conv1_bias_scale);
    golden::maxpool2d<HEIGHT, WIDTH, BASE>(relu_a, pooled_a);

    static act_t relu_b[HALF_H][HALF_W][2 * BASE];
    static act_t pooled_b[QUARTER_H][QUARTER_W][2 * BASE], upsampled_b[HALF_H][HALF_W][2 * BASE];
    golden::conv2d<HALF_H, HALF_W, BASE, 2 * BASE, KERNEL, activation::relu>(
        pooled_a, conv2_weight, conv2_bias, relu_b, conv2_weight_scale, conv2_bias_scale);
    golden::maxpool2d<HALF_H, HALF_W, 2 * BASE>(relu_b, pooled_b);
    golden::upsample2d<QUARTER_H, QUARTER_W, 2 * BASE>(pooled_b, upsampled_b);

    static act_t relu_c[HALF_H][HALF_W][BASE], upsampled_c[HEIGHT][WIDTH][BASE];
    golden::conv2d<HALF_H, HALF_W, 2 * BASE, BASE, KERNEL, activation::relu>(
        upsampled_b, conv3_weight, conv3_bias, relu_c, conv3_weight_scale, conv3_bias_scale);
    golden::upsample2d<HALF_H, HALF_W, BASE>(relu_c, upsampled_c);

    // The scores: no ReLU.
    golden::conv2d<HEIGHT, WIDTH, BASE, CLASSES, KERNEL>(
        upsampled_c, conv4_weight, conv4_bias, dout, conv4_weight_scale, conv4_bias_scale);
}
