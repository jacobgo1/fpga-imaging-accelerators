#pragma once

#include <cstdint>
#include <cstring>
#include "specpw_unet_golden.hpp"   // the sizes: specpw_unet_small, _medium, _large
#include "maxpool_golden.hpp"
#include "nearest_neighbor_golden.hpp"
#include "concat_golden.hpp"

// The quantized SpecPW-UNet in integers, written as plain loops: the arithmetic
// an optimized kernel must match bit for bit (its testbench's exact reference),
// and the same as classify_fixed in tools/specpw_unet_model.py. Not for HLS.
//
// Formats, fraction bits passed in (the numpy model's IN_FRAC, ACT_FRAC, OUT_FRAC):
//  - din: the z-score * 2^in_frac, int8 values (held in int32);
//  - each conv sums act * q exactly in int64, plus its bias shifted up to the
//    sum's fraction bits (act frac + weight frac); then ReLU, round half up to
//    act_frac (add half, shift right) and saturate at act_max: int16 activations;
//  - maxpool, nearest upsampling and the skip's concat move integers as they are;
//  - the final 1x1 conv rounds to out_frac, no ReLU: int32 scores.
// Integer sums do not depend on the order they are added in, so any kernel
// that keeps every product and rounds at the same places gets the same answer.

// One conv's int8 weights and biases with their fracs (value = q * 2^-frac).
template<int OUT_CH, int IN_CH, int K>
struct qconv_golden {
    int8_t w[OUT_CH][IN_CH][K][K];
    int w_frac;
    int8_t b[OUT_CH];
    int b_frac;
};

template<int OUT_CH, int IN_CH, int K, typename QW, typename QB>
void load_qconv_golden(qconv_golden<OUT_CH, IN_CH, K> &layer, const QW &w, int w_frac, const QB &b, int b_frac) {
    static_assert(sizeof(QW) == sizeof(layer.w) && sizeof(QB) == sizeof(layer.b), "the weights differ in shape");
    std::memcpy(layer.w, &w, sizeof(layer.w));
    std::memcpy(layer.b, &b, sizeof(layer.b));
    layer.w_frac = w_frac;
    layer.b_frac = b_frac;
}

template<int IN_CH, int N_CLS, typename Size>
struct specpw_unet_qparams_golden {
    qconv_golden<Size::S1, IN_CH, 1> sp1;
    qconv_golden<Size::S2, Size::S1, 1> sp2;
    qconv_golden<Size::S3, Size::S2, 1> sp3;
    qconv_golden<Size::C1, Size::S3, 3> d1a;
    qconv_golden<Size::C1, Size::C1, 3> d1b;
    qconv_golden<Size::C2, Size::C1, 3> d2a;
    qconv_golden<Size::C2, Size::C2, 3> d2b;
    qconv_golden<Size::C2, Size::C2, 3> up1;
    qconv_golden<Size::C3, Size::C2, 3> d3a;
    qconv_golden<Size::C3, Size::C3, 3> d3b;
    qconv_golden<Size::C3, Size::C3, 3> up2;
    qconv_golden<Size::C4, Size::C3 + Size::C1, 3> d4a;
    qconv_golden<Size::C4, Size::C4, 3> d4b;
    qconv_golden<N_CLS, Size::C4, 1> fin;
};

// Fills a specpw_unet_qparams_golden from a weights header's namespace
// (weights/quantized/sp_unet_SIZE/sp_unet_SIZE_int8.hpp).
#define SPECPW_UNET_QCONV(dst, ns, layer) \
    load_qconv_golden(dst, ns::layer##_weight, ns::layer##_weight_frac, ns::layer##_bias, ns::layer##_bias_frac)
#define SPECPW_UNET_LOAD_QPARAMS(p, ns) do { \
    SPECPW_UNET_QCONV((p).sp1, ns, convsp1); SPECPW_UNET_QCONV((p).sp2, ns, convsp2); \
    SPECPW_UNET_QCONV((p).sp3, ns, convsp3); \
    SPECPW_UNET_QCONV((p).d1a, ns, double1_block_0); SPECPW_UNET_QCONV((p).d1b, ns, double1_block_3); \
    SPECPW_UNET_QCONV((p).d2a, ns, double2_block_0); SPECPW_UNET_QCONV((p).d2b, ns, double2_block_3); \
    SPECPW_UNET_QCONV((p).up1, ns, up1_1); \
    SPECPW_UNET_QCONV((p).d3a, ns, double3_block_0); SPECPW_UNET_QCONV((p).d3b, ns, double3_block_3); \
    SPECPW_UNET_QCONV((p).up2, ns, up2_1); \
    SPECPW_UNET_QCONV((p).d4a, ns, double4_block_0); SPECPW_UNET_QCONV((p).d4b, ns, double4_block_3); \
    SPECPW_UNET_QCONV((p).fin, ns, final); } while (0)

// s * 2^-shift, rounded half up; a negative shift is exact (shifts left).
inline int64_t round_shift_golden(int64_t s, int shift) {
    if (shift <= 0)
        return s * (int64_t(1) << -shift);
    return (s + (int64_t(1) << (shift - 1))) >> shift;
}

// Same-padded KxK conv, then (RELU) ReLU and saturation at act_max, or (not
// RELU) only the rounding: the final layer's scores.
template<int H, int W, int IN_CH, int OUT_CH, int K, bool RELU>
void conv2d_same_fixed_golden(const int32_t din[H][W][IN_CH],
                              const qconv_golden<OUT_CH, IN_CH, K> &layer,
                              int in_frac, int out_frac, int32_t act_max,
                              int32_t dout[H][W][OUT_CH]) {
    constexpr int PAD = (K - 1) / 2;
    const int sum_frac = in_frac + layer.w_frac;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            for (int oc = 0; oc < OUT_CH; oc++) {
                int64_t sum = (int64_t)layer.b[oc] * (int64_t(1) << (sum_frac - layer.b_frac));
                for (int ic = 0; ic < IN_CH; ic++)
                    for (int ki = 0; ki < K; ki++)
                        for (int kj = 0; kj < K; kj++) {
                            const int y = i + ki - PAD, x = j + kj - PAD;
                            if (y < 0 || y >= H || x < 0 || x >= W) continue;
                            sum += (int64_t)din[y][x][ic] * layer.w[oc][ic][ki][kj];
                        }
                int64_t v = round_shift_golden(sum, sum_frac - out_frac);
                if (RELU)
                    v = v < 0 ? 0 : (v > act_max ? act_max : v);
                dout[i][j][oc] = (int32_t)v;
            }
}

template<int H, int W, int IN_CH, int N_CLS, typename Size>
void specpw_unet_fixed_golden(const int32_t din[H][W][IN_CH],
                              const specpw_unet_qparams_golden<IN_CH, N_CLS, Size> &p,
                              int in_frac, int act_frac, int out_frac, int32_t act_max,
                              int32_t dout[H][W][N_CLS]) {
    static_assert(H % 4 == 0 && W % 4 == 0, "H and W must be divisible by 4 (two maxpools)");
    constexpr int S1 = Size::S1, S2 = Size::S2, S3 = Size::S3;
    constexpr int C1 = Size::C1, C2 = Size::C2, C3 = Size::C3, C4 = Size::C4;
    constexpr int H2 = H / 2, W2 = W / 2, H4 = H / 4, W4 = W / 4;
    const int a = act_frac;

    static int32_t s1[H][W][S1], s2[H][W][S2], s3[H][W][S3];
    static int32_t d1[H][W][C1], skip[H][W][C1], pool1[H2][W2][C1];
    static int32_t d2[H2][W2][C2], enc2[H2][W2][C2], pool2[H4][W4][C2];
    static int32_t up1_in[H2][W2][C2], up1[H2][W2][C2];
    static int32_t d3[H2][W2][C3], dec3[H2][W2][C3];
    static int32_t up2_in[H][W][C3], up2[H][W][C3], cat[H][W][C3 + C1];
    static int32_t d4[H][W][C4], dec4[H][W][C4];

    conv2d_same_fixed_golden<H, W, IN_CH, S1, 1, true>(din, p.sp1, in_frac, a, act_max, s1);
    conv2d_same_fixed_golden<H, W, S1, S2, 1, true>(s1, p.sp2, a, a, act_max, s2);
    conv2d_same_fixed_golden<H, W, S2, S3, 1, true>(s2, p.sp3, a, a, act_max, s3);

    conv2d_same_fixed_golden<H, W, S3, C1, 3, true>(s3, p.d1a, a, a, act_max, d1);
    conv2d_same_fixed_golden<H, W, C1, C1, 3, true>(d1, p.d1b, a, a, act_max, skip);
    maxpool_golden<int32_t, H, W, C1, 2>(skip, pool1);
    conv2d_same_fixed_golden<H2, W2, C1, C2, 3, true>(pool1, p.d2a, a, a, act_max, d2);
    conv2d_same_fixed_golden<H2, W2, C2, C2, 3, true>(d2, p.d2b, a, a, act_max, enc2);
    maxpool_golden<int32_t, H2, W2, C2, 2>(enc2, pool2);

    nearest_neighbor_golden<int32_t, H4, W4, C2, 2>(pool2, up1_in);
    conv2d_same_fixed_golden<H2, W2, C2, C2, 3, true>(up1_in, p.up1, a, a, act_max, up1);
    conv2d_same_fixed_golden<H2, W2, C2, C3, 3, true>(up1, p.d3a, a, a, act_max, d3);
    conv2d_same_fixed_golden<H2, W2, C3, C3, 3, true>(d3, p.d3b, a, a, act_max, dec3);
    nearest_neighbor_golden<int32_t, H2, W2, C3, 2>(dec3, up2_in);
    conv2d_same_fixed_golden<H, W, C3, C3, 3, true>(up2_in, p.up2, a, a, act_max, up2);

    concat_golden<int32_t, H, W, C3, C1>(up2, skip, cat);
    conv2d_same_fixed_golden<H, W, C3 + C1, C4, 3, true>(cat, p.d4a, a, a, act_max, d4);
    conv2d_same_fixed_golden<H, W, C4, C4, 3, true>(d4, p.d4b, a, a, act_max, dec4);
    conv2d_same_fixed_golden<H, W, C4, N_CLS, 1, false>(dec4, p.fin, a, out_frac, act_max, dout);
}
