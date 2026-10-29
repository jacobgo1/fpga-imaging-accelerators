#pragma once

#include "spectral_stem_golden.hpp"
#include "conv_block_golden.hpp"
#include "up_block_golden.hpp"
#include "maxpool_golden.hpp"
#include "concat_golden.hpp"
#include "conv2d_same_golden.hpp"

// SpecPW-UNet as trained, in float: a spectral stem of 1x1 convs, then a
// two-level U-Net with one skip connection.
//
//   x    = stem(x)                                 BANDS -> STEM1 -> STEM2 -> STEM3
//   skip = double1(x)                              full size
//   x    = maxpool(double2(maxpool(skip)))         a quarter
//   x    = double3(up1(x))                         half
//   x    = up2(x)                                  full size
//   x    = double4(cat([x, skip]))
//   out  = final(x)                                1x1 conv to CLASSES
//
// The params struct is the state_dict's tensors in order, so a checkpoint loads
// as one block of floats. HEIGHT and WIDTH must be multiples of 4.
struct specpw_unet_small  { static constexpr int STEM1 = 24, STEM2 = 12, STEM3 = 12, DOUBLE1 = 12, DOUBLE2 = 24, DOUBLE3 = 12, DOUBLE4 = 8; };
struct specpw_unet_medium { static constexpr int STEM1 = 24, STEM2 = 16, STEM3 = 12, DOUBLE1 = 24, DOUBLE2 = 48, DOUBLE3 = 24, DOUBLE4 = 12; };
struct specpw_unet_large  { static constexpr int STEM1 = 32, STEM2 = 16, STEM3 = 16, DOUBLE1 = 32, DOUBLE2 = 64, DOUBLE3 = 32, DOUBLE4 = 16; };

template<typename data_t, int BANDS, int CLASSES, typename Size>
struct specpw_unet_params_golden {
    spectral_stem_params_golden<data_t, BANDS, Size::STEM1, Size::STEM2, Size::STEM3> stem;
    conv_block_params_golden<data_t, Size::STEM3, Size::DOUBLE1> double1;
    conv_block_params_golden<data_t, Size::DOUBLE1, Size::DOUBLE2> double2;
    up_block_params_golden<data_t, Size::DOUBLE2, Size::DOUBLE2> up1;
    conv_block_params_golden<data_t, Size::DOUBLE2, Size::DOUBLE3> double3;
    up_block_params_golden<data_t, Size::DOUBLE3, Size::DOUBLE3> up2;
    conv_block_params_golden<data_t, Size::DOUBLE3 + Size::DOUBLE1, Size::DOUBLE4> double4;
    data_t final_weight[CLASSES][Size::DOUBLE4][1][1];
    data_t final_bias[CLASSES];
};

template<typename data_t, int HEIGHT, int WIDTH, int BANDS, int CLASSES, typename Size>
void specpw_unet_golden(const data_t din[HEIGHT][WIDTH][BANDS],
                        const specpw_unet_params_golden<data_t, BANDS, CLASSES, Size> &params,
                        data_t eps,
                        data_t dout[HEIGHT][WIDTH][CLASSES]) {
    constexpr int D1 = Size::DOUBLE1, D2 = Size::DOUBLE2, D3 = Size::DOUBLE3, D4 = Size::DOUBLE4;
    constexpr int HALF_H = HEIGHT / 2, HALF_W = WIDTH / 2, QUARTER_H = HEIGHT / 4, QUARTER_W = WIDTH / 4;

    static data_t stem[HEIGHT][WIDTH][Size::STEM3];
    static data_t skip[HEIGHT][WIDTH][D1];
    static data_t pooled1[HALF_H][HALF_W][D1];
    static data_t double2[HALF_H][HALF_W][D2];
    static data_t pooled2[QUARTER_H][QUARTER_W][D2];
    static data_t up1[HALF_H][HALF_W][D2];
    static data_t double3[HALF_H][HALF_W][D3];
    static data_t up2[HEIGHT][WIDTH][D3];
    static data_t joined[HEIGHT][WIDTH][D3 + D1];
    static data_t double4[HEIGHT][WIDTH][D4];

    spectral_stem_golden<data_t, HEIGHT, WIDTH, BANDS, Size::STEM1, Size::STEM2, Size::STEM3>(din, params.stem, eps, stem);

    conv_block_golden<data_t, HEIGHT, WIDTH, Size::STEM3, D1>(stem, params.double1, eps, skip);
    maxpool_golden<data_t, HEIGHT, WIDTH, D1>(skip, pooled1);
    conv_block_golden<data_t, HALF_H, HALF_W, D1, D2>(pooled1, params.double2, eps, double2);
    maxpool_golden<data_t, HALF_H, HALF_W, D2>(double2, pooled2);

    up_block_golden<data_t, QUARTER_H, QUARTER_W, D2, D2>(pooled2, params.up1, eps, up1);
    conv_block_golden<data_t, HALF_H, HALF_W, D2, D3>(up1, params.double3, eps, double3);
    up_block_golden<data_t, HALF_H, HALF_W, D3, D3>(double3, params.up2, eps, up2);

    concat_golden<data_t, HEIGHT, WIDTH, D3, D1>(up2, skip, joined);
    conv_block_golden<data_t, HEIGHT, WIDTH, D3 + D1, D4>(joined, params.double4, eps, double4);
    conv2d_same_golden<data_t, HEIGHT, WIDTH, D4, CLASSES, 1>(double4, params.final_weight, params.final_bias, dout);
}
