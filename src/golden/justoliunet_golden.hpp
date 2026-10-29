#pragma once

#include "quant_scale.hpp"
#include "conv1d_quant_golden.hpp"
#include "relu1d_golden.hpp"
#include "maxpool1d_golden.hpp"
#include "flatten_golden.hpp"
#include "dense_quant_golden.hpp"

// The quantized 1D-Justo-LiuNet, one pixel: its spectrum in, 3 class scores out.
// Four blocks of conv1d (no padding), ReLU, maxpool by 2, then a dense layer:
// 110 bands -> 105 -> 52 -> 47 -> 23 -> 18 -> 9 -> 4 -> 2 positions, 24 x 2 = 48
// values into the dense layer. justoliunet_bn's BatchNorm is folded into conv1.

constexpr int JLN_BANDS = 110;           // raw bands 8-117, z-scored
constexpr int JLN_KERNEL_SIZE = 6;
constexpr int JLN_BASE_CHANNELS = 6;     // conv1's; conv2-4 have 2x, 3x, 4x
constexpr int JLN_CLASSES = 3;           // 0 cloud, 1 land, 2 sea

// The spectrum's length after one block, and after all four.
constexpr int justoliunet_block_length(int length, int kernel) { return (length - kernel + 1) / 2; }
constexpr int justoliunet_flat_length(int length, int kernel) {
    return justoliunet_block_length(justoliunet_block_length(justoliunet_block_length(
               justoliunet_block_length(length, kernel), kernel), kernel), kernel);
}

template<typename act_t, typename weight_t, int BANDS, int BASE, int CLASSES, int KERNEL>
void justoliunet_golden(
    const act_t din[BANDS][1],
    const weight_t conv1_weight[BASE][1][KERNEL],          act_t conv1_weight_scale,
    const weight_t conv1_bias[BASE],                       act_t conv1_bias_scale,
    const weight_t conv2_weight[2 * BASE][BASE][KERNEL],     act_t conv2_weight_scale,
    const weight_t conv2_bias[2 * BASE],                   act_t conv2_bias_scale,
    const weight_t conv3_weight[3 * BASE][2 * BASE][KERNEL], act_t conv3_weight_scale,
    const weight_t conv3_bias[3 * BASE],                   act_t conv3_bias_scale,
    const weight_t conv4_weight[4 * BASE][3 * BASE][KERNEL], act_t conv4_weight_scale,
    const weight_t conv4_bias[4 * BASE],                   act_t conv4_bias_scale,
    const weight_t dense_weight[CLASSES][4 * BASE * justoliunet_flat_length(BANDS, KERNEL)], act_t dense_weight_scale,
    const weight_t dense_bias[CLASSES],                    act_t dense_bias_scale,
    act_t dout[CLASSES])
{
    // Lengths after each conv and each pool.
    constexpr int CONV1 = BANDS - KERNEL + 1, POOL1 = CONV1 / 2;
    constexpr int CONV2 = POOL1 - KERNEL + 1, POOL2 = CONV2 / 2;
    constexpr int CONV3 = POOL2 - KERNEL + 1, POOL3 = CONV3 / 2;
    constexpr int CONV4 = POOL3 - KERNEL + 1, POOL4 = CONV4 / 2;
    constexpr int FLAT = 4 * BASE * POOL4;

    act_t conv1[CONV1][BASE], relu1[CONV1][BASE], pool1[POOL1][BASE];
    conv1d_quant_golden<act_t, weight_t, BANDS, 1, BASE, KERNEL>(
        din, conv1_weight, conv1_weight_scale, conv1_bias, conv1_bias_scale, conv1);
    relu1d_golden<act_t, CONV1, BASE>(conv1, relu1);
    maxpool1d_golden<act_t, CONV1, BASE, 2>(relu1, pool1);

    act_t conv2[CONV2][2 * BASE], relu2[CONV2][2 * BASE], pool2[POOL2][2 * BASE];
    conv1d_quant_golden<act_t, weight_t, POOL1, BASE, 2 * BASE, KERNEL>(
        pool1, conv2_weight, conv2_weight_scale, conv2_bias, conv2_bias_scale, conv2);
    relu1d_golden<act_t, CONV2, 2 * BASE>(conv2, relu2);
    maxpool1d_golden<act_t, CONV2, 2 * BASE, 2>(relu2, pool2);

    act_t conv3[CONV3][3 * BASE], relu3[CONV3][3 * BASE], pool3[POOL3][3 * BASE];
    conv1d_quant_golden<act_t, weight_t, POOL2, 2 * BASE, 3 * BASE, KERNEL>(
        pool2, conv3_weight, conv3_weight_scale, conv3_bias, conv3_bias_scale, conv3);
    relu1d_golden<act_t, CONV3, 3 * BASE>(conv3, relu3);
    maxpool1d_golden<act_t, CONV3, 3 * BASE, 2>(relu3, pool3);

    act_t conv4[CONV4][4 * BASE], relu4[CONV4][4 * BASE], pool4[POOL4][4 * BASE];
    conv1d_quant_golden<act_t, weight_t, POOL3, 3 * BASE, 4 * BASE, KERNEL>(
        pool3, conv4_weight, conv4_weight_scale, conv4_bias, conv4_bias_scale, conv4);
    relu1d_golden<act_t, CONV4, 4 * BASE>(conv4, relu4);
    maxpool1d_golden<act_t, CONV4, 4 * BASE, 2>(relu4, pool4);

    act_t flat[FLAT];
    flatten_golden<act_t, POOL4, 4 * BASE>(pool4, flat);
    dense_quant_golden<act_t, weight_t, FLAT, CLASSES>(
        flat, dense_weight, dense_weight_scale, dense_bias, dense_bias_scale, dout);
}
