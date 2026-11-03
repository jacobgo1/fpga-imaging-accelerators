#pragma once

#include "../golden/batchnorm.hpp"
#include "../../src/models/SpecPW-UNet/specpw_unet.hpp"   // the sizes

// SpecPW-UNet as trained, in float, with its BatchNorms: the testbenches' reference
// for the golden kernels (src/models/SpecPW-UNet/specpw_unet.hpp has the network).
// The params structs are the state_dict's tensors in order, so a checkpoint loads
// as one block of floats.

// One BatchNorm's parameters.
template<typename data_t, int CHANNELS>
struct batchnorm_params_golden {
    data_t weight[CHANNELS];
    data_t bias[CHANNELS];
    data_t running_mean[CHANNELS];
    data_t running_var[CHANNELS];
};

template<int CHANNELS, golden::activation A, typename data_t>
void batchnorm_golden(data_t *data, int count, const batchnorm_params_golden<data_t, CHANNELS> &bn, data_t eps) {
    golden::batchnorm<CHANNELS, A>(data, count, bn.weight, bn.bias, bn.running_mean, bn.running_var, eps);
}

// The spectral stem: relu(bnsp1(convsp1(x))), relu(convsp2(x)), relu(convsp3(x)).
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
    using golden::activation;
    static data_t relu1[HEIGHT][WIDTH][STEM1], relu2[HEIGHT][WIDTH][STEM2];
    golden::conv2d<HEIGHT, WIDTH, IN_CHANNELS, STEM1, 1>(din, params.conv1_weight, params.conv1_bias, relu1);
    batchnorm_golden<STEM1, activation::relu>(&relu1[0][0][0], HEIGHT * WIDTH, params.bn1, eps);
    golden::conv2d<HEIGHT, WIDTH, STEM1, STEM2, 1, activation::relu>(relu1, params.conv2_weight, params.conv2_bias, relu2);
    golden::conv2d<HEIGHT, WIDTH, STEM2, STEM3, 1, activation::relu>(relu2, params.conv3_weight, params.conv3_bias, dout);
}

// A U-Net's double conv: two times 3x3 conv (no bias), BatchNorm, ReLU.
template<typename data_t, int IN_CHANNELS, int OUT_CHANNELS>
struct conv_block_params_golden {
    data_t conv1_weight[OUT_CHANNELS][IN_CHANNELS][3][3];
    batchnorm_params_golden<data_t, OUT_CHANNELS> bn1;
    data_t conv2_weight[OUT_CHANNELS][OUT_CHANNELS][3][3];
    batchnorm_params_golden<data_t, OUT_CHANNELS> bn2;
};

template<typename data_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void conv_block_golden(const data_t din[HEIGHT][WIDTH][IN_CHANNELS],
                       const conv_block_params_golden<data_t, IN_CHANNELS, OUT_CHANNELS> &params,
                       data_t eps,
                       data_t dout[HEIGHT][WIDTH][OUT_CHANNELS]) {
    using golden::activation;
    static const data_t no_bias[OUT_CHANNELS] = {};
    static data_t first[HEIGHT][WIDTH][OUT_CHANNELS];
    golden::conv2d<HEIGHT, WIDTH, IN_CHANNELS, OUT_CHANNELS, 3>(din, params.conv1_weight, no_bias, first);
    batchnorm_golden<OUT_CHANNELS, activation::relu>(&first[0][0][0], HEIGHT * WIDTH, params.bn1, eps);
    golden::conv2d<HEIGHT, WIDTH, OUT_CHANNELS, OUT_CHANNELS, 3>(first, params.conv2_weight, no_bias, dout);
    batchnorm_golden<OUT_CHANNELS, activation::relu>(&dout[0][0][0], HEIGHT * WIDTH, params.bn2, eps);
}

// A U-Net's up block: upsample by 2, 3x3 conv (no bias), BatchNorm, ReLU.
// HEIGHT and WIDTH are the input's.
template<typename data_t, int IN_CHANNELS, int OUT_CHANNELS>
struct up_block_params_golden {
    data_t conv_weight[OUT_CHANNELS][IN_CHANNELS][3][3];
    batchnorm_params_golden<data_t, OUT_CHANNELS> bn;
};

template<typename data_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void up_block_golden(const data_t din[HEIGHT][WIDTH][IN_CHANNELS],
                     const up_block_params_golden<data_t, IN_CHANNELS, OUT_CHANNELS> &params,
                     data_t eps,
                     data_t dout[2 * HEIGHT][2 * WIDTH][OUT_CHANNELS]) {
    static const data_t no_bias[OUT_CHANNELS] = {};
    static data_t upsampled[2 * HEIGHT][2 * WIDTH][IN_CHANNELS];
    golden::upsample2d<HEIGHT, WIDTH, IN_CHANNELS>(din, upsampled);
    golden::conv2d<2 * HEIGHT, 2 * WIDTH, IN_CHANNELS, OUT_CHANNELS, 3>(upsampled, params.conv_weight, no_bias, dout);
    batchnorm_golden<OUT_CHANNELS, golden::activation::relu>(&dout[0][0][0], 4 * HEIGHT * WIDTH, params.bn, eps);
}

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
    golden::maxpool2d<HEIGHT, WIDTH, D1>(skip, pooled1);
    conv_block_golden<data_t, HALF_H, HALF_W, D1, D2>(pooled1, params.double2, eps, double2);
    golden::maxpool2d<HALF_H, HALF_W, D2>(double2, pooled2);

    up_block_golden<data_t, QUARTER_H, QUARTER_W, D2, D2>(pooled2, params.up1, eps, up1);
    conv_block_golden<data_t, HALF_H, HALF_W, D2, D3>(up1, params.double3, eps, double3);
    up_block_golden<data_t, HALF_H, HALF_W, D3, D3>(double3, params.up2, eps, up2);

    golden::concat<HEIGHT, WIDTH, D3, D1>(up2, skip, joined);
    conv_block_golden<data_t, HEIGHT, WIDTH, D3 + D1, D4>(joined, params.double4, eps, double4);
    golden::conv2d<HEIGHT, WIDTH, D4, CLASSES, 1>(double4, params.final_weight, params.final_bias, dout);
}
