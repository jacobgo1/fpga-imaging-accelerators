#pragma once

#include "../../golden/layers.hpp"

// SpecPW-UNet, quantized: a spectral stem of 1x1 convs, then a two-level U-Net with
// one skip connection, every BatchNorm folded into the conv before it (the float
// network with them is tb/sp_unet_small/specpw_unet_golden.hpp):
//
//   x    = stem(x)                                 BANDS -> STEM1 -> STEM2 -> STEM3
//   skip = double1(x)                              full size
//   x    = maxpool(double2(maxpool(skip)))         a quarter
//   x    = double3(up1(x))                         half
//   x    = up2(x)                                  full size
//   x    = double4(cat([x, skip]))
//   out  = final(x)                                1x1 conv to CLASSES
//
// HEIGHT and WIDTH must be multiples of 4.
struct specpw_unet_small  { static constexpr int STEM1 = 24, STEM2 = 12, STEM3 = 12, DOUBLE1 = 12, DOUBLE2 = 24, DOUBLE3 = 12, DOUBLE4 = 8; };
struct specpw_unet_medium { static constexpr int STEM1 = 24, STEM2 = 16, STEM3 = 12, DOUBLE1 = 24, DOUBLE2 = 48, DOUBLE3 = 24, DOUBLE4 = 12; };
struct specpw_unet_large  { static constexpr int STEM1 = 32, STEM2 = 16, STEM3 = 16, DOUBLE1 = 32, DOUBLE2 = 64, DOUBLE3 = 32, DOUBLE4 = 16; };

// The spectral stem: three 1x1 convs with ReLU. Each pixel goes through all three on
// its own (a 1 x 1 map), so the input patch is never stored whole: din is the patch,
// HWC, at din[offset] (in a kernel, an AXI pointer).
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int STEM1, int STEM2, int STEM3>
void specpw_stem(
    const act_t *din, long offset,
    const weight_t conv1_weight[STEM1][IN_CHANNELS][1][1], act_t conv1_weight_scale,
    const weight_t conv1_bias[STEM1],                      act_t conv1_bias_scale,
    const weight_t conv2_weight[STEM2][STEM1][1][1],       act_t conv2_weight_scale,
    const weight_t conv2_bias[STEM2],                      act_t conv2_bias_scale,
    const weight_t conv3_weight[STEM3][STEM2][1][1],       act_t conv3_weight_scale,
    const weight_t conv3_bias[STEM3],                      act_t conv3_bias_scale,
    act_t dout[HEIGHT][WIDTH][STEM3])
{
    using golden::activation;
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++) {
            act_t spectrum[1][1][IN_CHANNELS], relu1[1][1][STEM1], relu2[1][1][STEM2], relu3[1][1][STEM3];
            const long first = offset + ((long)row * WIDTH + col) * IN_CHANNELS;
            for (int band = 0; band < IN_CHANNELS; band++)
                spectrum[0][0][band] = din[first + band];
            golden::conv2d<1, 1, IN_CHANNELS, STEM1, 1, activation::relu>(
                spectrum, conv1_weight, conv1_bias, relu1, conv1_weight_scale, conv1_bias_scale);
            golden::conv2d<1, 1, STEM1, STEM2, 1, activation::relu>(
                relu1, conv2_weight, conv2_bias, relu2, conv2_weight_scale, conv2_bias_scale);
            golden::conv2d<1, 1, STEM2, STEM3, 1, activation::relu>(
                relu2, conv3_weight, conv3_bias, relu3, conv3_weight_scale, conv3_bias_scale);
            for (int ch = 0; ch < STEM3; ch++)
                dout[row][col][ch] = relu3[0][0][ch];
        }
}

// A U-Net's double conv: two 3x3 convs, each with a ReLU.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void specpw_conv_block(
    const act_t din[HEIGHT][WIDTH][IN_CHANNELS],
    const weight_t conv1_weight[OUT_CHANNELS][IN_CHANNELS][3][3],  act_t conv1_weight_scale,
    const weight_t conv1_bias[OUT_CHANNELS],                       act_t conv1_bias_scale,
    const weight_t conv2_weight[OUT_CHANNELS][OUT_CHANNELS][3][3], act_t conv2_weight_scale,
    const weight_t conv2_bias[OUT_CHANNELS],                       act_t conv2_bias_scale,
    act_t dout[HEIGHT][WIDTH][OUT_CHANNELS])
{
    using golden::activation;
    // UltraRAM: BRAM is left for the weights.
    static act_t first[HEIGHT][WIDTH][OUT_CHANNELS];
    #pragma HLS BIND_STORAGE variable=first type=ram_2p impl=uram
    golden::conv2d<HEIGHT, WIDTH, IN_CHANNELS, OUT_CHANNELS, 3, activation::relu>(
        din, conv1_weight, conv1_bias, first, conv1_weight_scale, conv1_bias_scale);
    golden::conv2d<HEIGHT, WIDTH, OUT_CHANNELS, OUT_CHANNELS, 3, activation::relu>(
        first, conv2_weight, conv2_bias, dout, conv2_weight_scale, conv2_bias_scale);
}

// A U-Net's up block: upsample by 2, a 3x3 conv, ReLU. HEIGHT and WIDTH are the input's.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void specpw_up_block(
    const act_t din[HEIGHT][WIDTH][IN_CHANNELS],
    const weight_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], act_t weight_scale,
    const weight_t bias[OUT_CHANNELS],                      act_t bias_scale,
    act_t dout[2 * HEIGHT][2 * WIDTH][OUT_CHANNELS])
{
    static act_t upsampled[2 * HEIGHT][2 * WIDTH][IN_CHANNELS];
    #pragma HLS BIND_STORAGE variable=upsampled type=ram_2p impl=uram
    golden::upsample2d<HEIGHT, WIDTH, IN_CHANNELS>(din, upsampled);
    golden::conv2d<2 * HEIGHT, 2 * WIDTH, IN_CHANNELS, OUT_CHANNELS, 3, golden::activation::relu>(
        upsampled, weight, bias, dout, weight_scale, bias_scale);
}

// The network. Each weight comes with its scale, 2^-frac, the arguments in the
// state_dict's order. din is one patch, HWC, at din[offset]. The large maps are
// UltraRAM, so BRAM holds the weights.
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

    specpw_stem<act_t, weight_t, HEIGHT, WIDTH, BANDS, Size::STEM1, Size::STEM2, Size::STEM3>(
        din, offset,
        convsp1_weight, convsp1_weight_scale, convsp1_bias, convsp1_bias_scale,
        convsp2_weight, convsp2_weight_scale, convsp2_bias, convsp2_bias_scale,
        convsp3_weight, convsp3_weight_scale, convsp3_bias, convsp3_bias_scale, stem);

    specpw_conv_block<act_t, weight_t, HEIGHT, WIDTH, Size::STEM3, D1>(
        stem, double1_conv1_weight, double1_conv1_weight_scale, double1_conv1_bias, double1_conv1_bias_scale,
        double1_conv2_weight, double1_conv2_weight_scale, double1_conv2_bias, double1_conv2_bias_scale, skip);
    golden::maxpool2d<HEIGHT, WIDTH, D1>(skip, pooled1);
    specpw_conv_block<act_t, weight_t, HALF_H, HALF_W, D1, D2>(
        pooled1, double2_conv1_weight, double2_conv1_weight_scale, double2_conv1_bias, double2_conv1_bias_scale,
        double2_conv2_weight, double2_conv2_weight_scale, double2_conv2_bias, double2_conv2_bias_scale, double2);
    golden::maxpool2d<HALF_H, HALF_W, D2>(double2, pooled2);

    specpw_up_block<act_t, weight_t, QUARTER_H, QUARTER_W, D2, D2>(
        pooled2, up1_weight, up1_weight_scale, up1_bias, up1_bias_scale, up1);
    specpw_conv_block<act_t, weight_t, HALF_H, HALF_W, D2, D3>(
        up1, double3_conv1_weight, double3_conv1_weight_scale, double3_conv1_bias, double3_conv1_bias_scale,
        double3_conv2_weight, double3_conv2_weight_scale, double3_conv2_bias, double3_conv2_bias_scale, double3);
    specpw_up_block<act_t, weight_t, HALF_H, HALF_W, D3, D3>(
        double3, up2_weight, up2_weight_scale, up2_bias, up2_bias_scale, up2);

    golden::concat<HEIGHT, WIDTH, D3, D1>(up2, skip, joined);
    specpw_conv_block<act_t, weight_t, HEIGHT, WIDTH, D3 + D1, D4>(
        joined, double4_conv1_weight, double4_conv1_weight_scale, double4_conv1_bias, double4_conv1_bias_scale,
        double4_conv2_weight, double4_conv2_weight_scale, double4_conv2_bias, double4_conv2_bias_scale, double4);

    golden::conv2d<HEIGHT, WIDTH, D4, CLASSES, 1>(
        double4, final_weight, final_bias, dout, final_weight_scale, final_bias_scale);
}
