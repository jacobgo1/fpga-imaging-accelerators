#pragma once

#include "quant_scale.hpp"
#include "specpw_unet_golden.hpp"   // the sizes
#include "spectral_stem_quant_golden.hpp"
#include "conv_block_quant_golden.hpp"
#include "up_block_quant_golden.hpp"
#include "conv2d_same_quant_golden.hpp"
#include "maxpool_golden.hpp"
#include "concat_golden.hpp"

// The quantized SpecPW-UNet: specpw_unet_golden.hpp with every BatchNorm folded
// into the conv before it and int8 weights. Each weight comes with its scale,
// 2^-frac, the arguments in the state_dict's order. din is one patch, HWC, at
// din[offset]. The large maps are UltraRAM, so BRAM holds the weights.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int BANDS, int CLASSES, typename Size>
void specpw_unet_quant_golden(
    const act_t *din, long offset,
    const weight_t convsp1_weight[Size::STEM1][BANDS][1][1],          act_t convsp1_weight_scale,
    const weight_t convsp1_bias[Size::STEM1],                         act_t convsp1_bias_scale,
    const weight_t convsp2_weight[Size::STEM2][Size::STEM1][1][1],    act_t convsp2_weight_scale,
    const weight_t convsp2_bias[Size::STEM2],                         act_t convsp2_bias_scale,
    const weight_t convsp3_weight[Size::STEM3][Size::STEM2][1][1],    act_t convsp3_weight_scale,
    const weight_t convsp3_bias[Size::STEM3],                         act_t convsp3_bias_scale,
    const weight_t double1_conv1_weight[Size::DOUBLE1][Size::STEM3][3][3],   act_t double1_conv1_weight_scale,
    const weight_t double1_conv1_bias[Size::DOUBLE1],                        act_t double1_conv1_bias_scale,
    const weight_t double1_conv2_weight[Size::DOUBLE1][Size::DOUBLE1][3][3], act_t double1_conv2_weight_scale,
    const weight_t double1_conv2_bias[Size::DOUBLE1],                        act_t double1_conv2_bias_scale,
    const weight_t double2_conv1_weight[Size::DOUBLE2][Size::DOUBLE1][3][3], act_t double2_conv1_weight_scale,
    const weight_t double2_conv1_bias[Size::DOUBLE2],                        act_t double2_conv1_bias_scale,
    const weight_t double2_conv2_weight[Size::DOUBLE2][Size::DOUBLE2][3][3], act_t double2_conv2_weight_scale,
    const weight_t double2_conv2_bias[Size::DOUBLE2],                        act_t double2_conv2_bias_scale,
    const weight_t up1_weight[Size::DOUBLE2][Size::DOUBLE2][3][3],           act_t up1_weight_scale,
    const weight_t up1_bias[Size::DOUBLE2],                                  act_t up1_bias_scale,
    const weight_t double3_conv1_weight[Size::DOUBLE3][Size::DOUBLE2][3][3], act_t double3_conv1_weight_scale,
    const weight_t double3_conv1_bias[Size::DOUBLE3],                        act_t double3_conv1_bias_scale,
    const weight_t double3_conv2_weight[Size::DOUBLE3][Size::DOUBLE3][3][3], act_t double3_conv2_weight_scale,
    const weight_t double3_conv2_bias[Size::DOUBLE3],                        act_t double3_conv2_bias_scale,
    const weight_t up2_weight[Size::DOUBLE3][Size::DOUBLE3][3][3],           act_t up2_weight_scale,
    const weight_t up2_bias[Size::DOUBLE3],                                  act_t up2_bias_scale,
    const weight_t double4_conv1_weight[Size::DOUBLE4][Size::DOUBLE3 + Size::DOUBLE1][3][3], act_t double4_conv1_weight_scale,
    const weight_t double4_conv1_bias[Size::DOUBLE4],                        act_t double4_conv1_bias_scale,
    const weight_t double4_conv2_weight[Size::DOUBLE4][Size::DOUBLE4][3][3], act_t double4_conv2_weight_scale,
    const weight_t double4_conv2_bias[Size::DOUBLE4],                        act_t double4_conv2_bias_scale,
    const weight_t final_weight[CLASSES][Size::DOUBLE4][1][1],               act_t final_weight_scale,
    const weight_t final_bias[CLASSES],                                      act_t final_bias_scale,
    act_t dout[HEIGHT][WIDTH][CLASSES])
{
    constexpr int D1 = Size::DOUBLE1, D2 = Size::DOUBLE2, D3 = Size::DOUBLE3, D4 = Size::DOUBLE4;
    constexpr int HALF_H = HEIGHT / 2, HALF_W = WIDTH / 2, QUARTER_H = HEIGHT / 4, QUARTER_W = WIDTH / 4;

    static act_t stem[HEIGHT][WIDTH][Size::STEM3];
    static act_t skip[HEIGHT][WIDTH][D1];
    #pragma HLS BIND_STORAGE variable=skip type=ram_2p impl=uram
    static act_t pooled1[HALF_H][HALF_W][D1];
    static act_t double2[HALF_H][HALF_W][D2];
    #pragma HLS BIND_STORAGE variable=double2 type=ram_2p impl=uram
    static act_t pooled2[QUARTER_H][QUARTER_W][D2];
    static act_t up1[HALF_H][HALF_W][D2];
    #pragma HLS BIND_STORAGE variable=up1 type=ram_2p impl=uram
    static act_t double3[HALF_H][HALF_W][D3];
    static act_t up2[HEIGHT][WIDTH][D3];
    #pragma HLS BIND_STORAGE variable=up2 type=ram_2p impl=uram
    static act_t joined[HEIGHT][WIDTH][D3 + D1];
    #pragma HLS BIND_STORAGE variable=joined type=ram_2p impl=uram
    static act_t double4[HEIGHT][WIDTH][D4];
    #pragma HLS BIND_STORAGE variable=double4 type=ram_2p impl=uram

    spectral_stem_quant_golden<act_t, weight_t, HEIGHT, WIDTH, BANDS, Size::STEM1, Size::STEM2, Size::STEM3>(
        din, offset,
        convsp1_weight, convsp1_weight_scale, convsp1_bias, convsp1_bias_scale,
        convsp2_weight, convsp2_weight_scale, convsp2_bias, convsp2_bias_scale,
        convsp3_weight, convsp3_weight_scale, convsp3_bias, convsp3_bias_scale, stem);

    conv_block_quant_golden<act_t, weight_t, HEIGHT, WIDTH, Size::STEM3, D1>(
        stem, double1_conv1_weight, double1_conv1_weight_scale, double1_conv1_bias, double1_conv1_bias_scale,
        double1_conv2_weight, double1_conv2_weight_scale, double1_conv2_bias, double1_conv2_bias_scale, skip);
    maxpool_golden<act_t, HEIGHT, WIDTH, D1>(skip, pooled1);
    conv_block_quant_golden<act_t, weight_t, HALF_H, HALF_W, D1, D2>(
        pooled1, double2_conv1_weight, double2_conv1_weight_scale, double2_conv1_bias, double2_conv1_bias_scale,
        double2_conv2_weight, double2_conv2_weight_scale, double2_conv2_bias, double2_conv2_bias_scale, double2);
    maxpool_golden<act_t, HALF_H, HALF_W, D2>(double2, pooled2);

    up_block_quant_golden<act_t, weight_t, QUARTER_H, QUARTER_W, D2, D2>(
        pooled2, up1_weight, up1_weight_scale, up1_bias, up1_bias_scale, up1);
    conv_block_quant_golden<act_t, weight_t, HALF_H, HALF_W, D2, D3>(
        up1, double3_conv1_weight, double3_conv1_weight_scale, double3_conv1_bias, double3_conv1_bias_scale,
        double3_conv2_weight, double3_conv2_weight_scale, double3_conv2_bias, double3_conv2_bias_scale, double3);
    up_block_quant_golden<act_t, weight_t, HALF_H, HALF_W, D3, D3>(
        double3, up2_weight, up2_weight_scale, up2_bias, up2_bias_scale, up2);

    concat_golden<act_t, HEIGHT, WIDTH, D3, D1>(up2, skip, joined);
    conv_block_quant_golden<act_t, weight_t, HEIGHT, WIDTH, D3 + D1, D4>(
        joined, double4_conv1_weight, double4_conv1_weight_scale, double4_conv1_bias, double4_conv1_bias_scale,
        double4_conv2_weight, double4_conv2_weight_scale, double4_conv2_bias, double4_conv2_bias_scale, double4);

    conv2d_same_quant_golden<act_t, weight_t, HEIGHT, WIDTH, D4, CLASSES, 1>(
        double4, final_weight, final_weight_scale, final_bias, final_bias_scale, dout);
}
