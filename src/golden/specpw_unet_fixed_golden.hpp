#pragma once

#include <cstdint>
#include <cstring>
#include "specpw_unet_golden.hpp"   // the sizes
#include "maxpool_golden.hpp"
#include "upsample_golden.hpp"
#include "concat_golden.hpp"

// The quantized SpecPW-UNet in integers, as plain loops: what an optimized
// kernel must match bit for bit, and the same as classify_fixed in
// tools/specpw_unet_model.py. Not for HLS.
//  - din: the z-scores with in_frac fraction bits, int8 values;
//  - each conv sums exactly in int64, its bias shifted up to the sum's fraction
//    bits; then it rounds half up to act_frac, ReLU, saturates at act_max;
//  - the final 1x1 conv rounds to out_frac, no ReLU: the int32 scores.

// One conv's int8 weights and bias, value = q * 2^-frac.
template<int OUT_CHANNELS, int IN_CHANNELS, int KERNEL>
struct quantized_conv_golden {
    int8_t weight[OUT_CHANNELS][IN_CHANNELS][KERNEL][KERNEL];
    int weight_frac;
    int8_t bias[OUT_CHANNELS];
    int bias_frac;
};

template<int OUT_CHANNELS, int IN_CHANNELS, int KERNEL, typename Weight, typename Bias>
void load_quantized_conv_golden(quantized_conv_golden<OUT_CHANNELS, IN_CHANNELS, KERNEL> &conv,
                                const Weight &weight, int weight_frac, const Bias &bias, int bias_frac) {
    std::memcpy(conv.weight, &weight, sizeof(conv.weight));
    std::memcpy(conv.bias, &bias, sizeof(conv.bias));
    conv.weight_frac = weight_frac;
    conv.bias_frac = bias_frac;
}

template<int BANDS, int CLASSES, typename Size>
struct specpw_unet_quantized_params_golden {
    quantized_conv_golden<Size::STEM1, BANDS, 1> convsp1;
    quantized_conv_golden<Size::STEM2, Size::STEM1, 1> convsp2;
    quantized_conv_golden<Size::STEM3, Size::STEM2, 1> convsp3;
    quantized_conv_golden<Size::DOUBLE1, Size::STEM3, 3> double1_conv1;
    quantized_conv_golden<Size::DOUBLE1, Size::DOUBLE1, 3> double1_conv2;
    quantized_conv_golden<Size::DOUBLE2, Size::DOUBLE1, 3> double2_conv1;
    quantized_conv_golden<Size::DOUBLE2, Size::DOUBLE2, 3> double2_conv2;
    quantized_conv_golden<Size::DOUBLE2, Size::DOUBLE2, 3> up1;
    quantized_conv_golden<Size::DOUBLE3, Size::DOUBLE2, 3> double3_conv1;
    quantized_conv_golden<Size::DOUBLE3, Size::DOUBLE3, 3> double3_conv2;
    quantized_conv_golden<Size::DOUBLE3, Size::DOUBLE3, 3> up2;
    quantized_conv_golden<Size::DOUBLE4, Size::DOUBLE3 + Size::DOUBLE1, 3> double4_conv1;
    quantized_conv_golden<Size::DOUBLE4, Size::DOUBLE4, 3> double4_conv2;
    quantized_conv_golden<CLASSES, Size::DOUBLE4, 1> final_conv;
};

// Fills the params from a weights header's namespace (weights/quantized/sp_unet_SIZE/).
#define SPECPW_UNET_LOAD_CONV(conv, weights, name) \
    load_quantized_conv_golden(conv, weights::name##_weight, weights::name##_weight_frac, \
                               weights::name##_bias, weights::name##_bias_frac)
#define SPECPW_UNET_LOAD_QUANTIZED_PARAMS(params, weights) do { \
    SPECPW_UNET_LOAD_CONV((params).convsp1, weights, convsp1); \
    SPECPW_UNET_LOAD_CONV((params).convsp2, weights, convsp2); \
    SPECPW_UNET_LOAD_CONV((params).convsp3, weights, convsp3); \
    SPECPW_UNET_LOAD_CONV((params).double1_conv1, weights, double1_block_0); \
    SPECPW_UNET_LOAD_CONV((params).double1_conv2, weights, double1_block_3); \
    SPECPW_UNET_LOAD_CONV((params).double2_conv1, weights, double2_block_0); \
    SPECPW_UNET_LOAD_CONV((params).double2_conv2, weights, double2_block_3); \
    SPECPW_UNET_LOAD_CONV((params).up1, weights, up1_1); \
    SPECPW_UNET_LOAD_CONV((params).double3_conv1, weights, double3_block_0); \
    SPECPW_UNET_LOAD_CONV((params).double3_conv2, weights, double3_block_3); \
    SPECPW_UNET_LOAD_CONV((params).up2, weights, up2_1); \
    SPECPW_UNET_LOAD_CONV((params).double4_conv1, weights, double4_block_0); \
    SPECPW_UNET_LOAD_CONV((params).double4_conv2, weights, double4_block_3); \
    SPECPW_UNET_LOAD_CONV((params).final_conv, weights, final); } while (0)

// value * 2^-shift, rounded half up; a negative shift shifts left, exactly.
inline int64_t round_shift_golden(int64_t value, int shift) {
    if (shift <= 0)
        return value * (int64_t(1) << -shift);
    return (value + (int64_t(1) << (shift - 1))) >> shift;
}

// Same-padded conv; with RELU also ReLU and saturation at act_max, without it
// only the rounding (the final layer).
template<int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL, bool RELU>
void conv2d_same_fixed_golden(const int32_t din[HEIGHT][WIDTH][IN_CHANNELS],
                              const quantized_conv_golden<OUT_CHANNELS, IN_CHANNELS, KERNEL> &conv,
                              int in_frac, int out_frac, int32_t act_max,
                              int32_t dout[HEIGHT][WIDTH][OUT_CHANNELS]) {
    constexpr int PAD = (KERNEL - 1) / 2;
    const int sum_frac = in_frac + conv.weight_frac;
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int out = 0; out < OUT_CHANNELS; out++) {
                int64_t sum = (int64_t)conv.bias[out] * (int64_t(1) << (sum_frac - conv.bias_frac));
                for (int in = 0; in < IN_CHANNELS; in++)
                    for (int ky = 0; ky < KERNEL; ky++)
                        for (int kx = 0; kx < KERNEL; kx++) {
                            const int y = row + ky - PAD, x = col + kx - PAD;
                            if (y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH)
                                sum += (int64_t)din[y][x][in] * conv.weight[out][in][ky][kx];
                        }
                int64_t value = round_shift_golden(sum, sum_frac - out_frac);
                if (RELU)
                    value = value < 0 ? 0 : (value > act_max ? act_max : value);
                dout[row][col][out] = (int32_t)value;
            }
}

template<int HEIGHT, int WIDTH, int BANDS, int CLASSES, typename Size>
void specpw_unet_fixed_golden(const int32_t din[HEIGHT][WIDTH][BANDS],
                              const specpw_unet_quantized_params_golden<BANDS, CLASSES, Size> &params,
                              int in_frac, int act_frac, int out_frac, int32_t act_max,
                              int32_t dout[HEIGHT][WIDTH][CLASSES]) {
    constexpr int S1 = Size::STEM1, S2 = Size::STEM2, S3 = Size::STEM3;
    constexpr int D1 = Size::DOUBLE1, D2 = Size::DOUBLE2, D3 = Size::DOUBLE3, D4 = Size::DOUBLE4;
    constexpr int HALF_H = HEIGHT / 2, HALF_W = WIDTH / 2, QUARTER_H = HEIGHT / 4, QUARTER_W = WIDTH / 4;
    const int act = act_frac;

    static int32_t stem1[HEIGHT][WIDTH][S1], stem2[HEIGHT][WIDTH][S2], stem3[HEIGHT][WIDTH][S3];
    static int32_t double1_first[HEIGHT][WIDTH][D1], skip[HEIGHT][WIDTH][D1], pooled1[HALF_H][HALF_W][D1];
    static int32_t double2_first[HALF_H][HALF_W][D2], double2[HALF_H][HALF_W][D2], pooled2[QUARTER_H][QUARTER_W][D2];
    static int32_t up1_input[HALF_H][HALF_W][D2], up1[HALF_H][HALF_W][D2];
    static int32_t double3_first[HALF_H][HALF_W][D3], double3[HALF_H][HALF_W][D3];
    static int32_t up2_input[HEIGHT][WIDTH][D3], up2[HEIGHT][WIDTH][D3], joined[HEIGHT][WIDTH][D3 + D1];
    static int32_t double4_first[HEIGHT][WIDTH][D4], double4[HEIGHT][WIDTH][D4];

    conv2d_same_fixed_golden<HEIGHT, WIDTH, BANDS, S1, 1, true>(din, params.convsp1, in_frac, act, act_max, stem1);
    conv2d_same_fixed_golden<HEIGHT, WIDTH, S1, S2, 1, true>(stem1, params.convsp2, act, act, act_max, stem2);
    conv2d_same_fixed_golden<HEIGHT, WIDTH, S2, S3, 1, true>(stem2, params.convsp3, act, act, act_max, stem3);

    conv2d_same_fixed_golden<HEIGHT, WIDTH, S3, D1, 3, true>(stem3, params.double1_conv1, act, act, act_max, double1_first);
    conv2d_same_fixed_golden<HEIGHT, WIDTH, D1, D1, 3, true>(double1_first, params.double1_conv2, act, act, act_max, skip);
    maxpool_golden<int32_t, HEIGHT, WIDTH, D1>(skip, pooled1);
    conv2d_same_fixed_golden<HALF_H, HALF_W, D1, D2, 3, true>(pooled1, params.double2_conv1, act, act, act_max, double2_first);
    conv2d_same_fixed_golden<HALF_H, HALF_W, D2, D2, 3, true>(double2_first, params.double2_conv2, act, act, act_max, double2);
    maxpool_golden<int32_t, HALF_H, HALF_W, D2>(double2, pooled2);

    upsample_golden<int32_t, QUARTER_H, QUARTER_W, D2>(pooled2, up1_input);
    conv2d_same_fixed_golden<HALF_H, HALF_W, D2, D2, 3, true>(up1_input, params.up1, act, act, act_max, up1);
    conv2d_same_fixed_golden<HALF_H, HALF_W, D2, D3, 3, true>(up1, params.double3_conv1, act, act, act_max, double3_first);
    conv2d_same_fixed_golden<HALF_H, HALF_W, D3, D3, 3, true>(double3_first, params.double3_conv2, act, act, act_max, double3);
    upsample_golden<int32_t, HALF_H, HALF_W, D3>(double3, up2_input);
    conv2d_same_fixed_golden<HEIGHT, WIDTH, D3, D3, 3, true>(up2_input, params.up2, act, act, act_max, up2);

    concat_golden<int32_t, HEIGHT, WIDTH, D3, D1>(up2, skip, joined);
    conv2d_same_fixed_golden<HEIGHT, WIDTH, D3 + D1, D4, 3, true>(joined, params.double4_conv1, act, act, act_max, double4_first);
    conv2d_same_fixed_golden<HEIGHT, WIDTH, D4, D4, 3, true>(double4_first, params.double4_conv2, act, act, act_max, double4);
    conv2d_same_fixed_golden<HEIGHT, WIDTH, D4, CLASSES, 1, false>(double4, params.final_conv, act, out_frac, act_max, dout);
}
