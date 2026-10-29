#pragma once

#include "quant_scale.hpp"
#include "conv2d_same_quant_golden.hpp"
#include "relu_golden.hpp"
#include "maxpool_golden.hpp"
#include "upsample_golden.hpp"

// The quantized 2D-JustoUNet: justounet2d_golden.hpp with the BatchNorms folded
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

    static act_t conv_a[HEIGHT][WIDTH][BASE], relu_a[HEIGHT][WIDTH][BASE], pooled_a[HALF_H][HALF_W][BASE];
    conv2d_same_quant_golden<act_t, weight_t, HEIGHT, WIDTH, BANDS, BASE, KERNEL>(
        din, conv1_weight, conv1_weight_scale, conv1_bias, conv1_bias_scale, conv_a);
    relu_golden<act_t, HEIGHT, WIDTH, BASE>(conv_a, relu_a);
    maxpool_golden<act_t, HEIGHT, WIDTH, BASE>(relu_a, pooled_a);

    static act_t conv_b[HALF_H][HALF_W][2 * BASE], relu_b[HALF_H][HALF_W][2 * BASE];
    static act_t pooled_b[QUARTER_H][QUARTER_W][2 * BASE], upsampled_b[HALF_H][HALF_W][2 * BASE];
    conv2d_same_quant_golden<act_t, weight_t, HALF_H, HALF_W, BASE, 2 * BASE, KERNEL>(
        pooled_a, conv2_weight, conv2_weight_scale, conv2_bias, conv2_bias_scale, conv_b);
    relu_golden<act_t, HALF_H, HALF_W, 2 * BASE>(conv_b, relu_b);
    maxpool_golden<act_t, HALF_H, HALF_W, 2 * BASE>(relu_b, pooled_b);
    upsample_golden<act_t, QUARTER_H, QUARTER_W, 2 * BASE>(pooled_b, upsampled_b);

    static act_t conv_c[HALF_H][HALF_W][BASE], relu_c[HALF_H][HALF_W][BASE], upsampled_c[HEIGHT][WIDTH][BASE];
    conv2d_same_quant_golden<act_t, weight_t, HALF_H, HALF_W, 2 * BASE, BASE, KERNEL>(
        upsampled_b, conv3_weight, conv3_weight_scale, conv3_bias, conv3_bias_scale, conv_c);
    relu_golden<act_t, HALF_H, HALF_W, BASE>(conv_c, relu_c);
    upsample_golden<act_t, HALF_H, HALF_W, BASE>(relu_c, upsampled_c);

    // The scores: no ReLU.
    conv2d_same_quant_golden<act_t, weight_t, HEIGHT, WIDTH, BASE, CLASSES, KERNEL>(
        upsampled_c, conv4_weight, conv4_weight_scale, conv4_bias, conv4_bias_scale, dout);
}
