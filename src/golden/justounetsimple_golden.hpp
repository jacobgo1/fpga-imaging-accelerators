#pragma once

#include "relu_golden.hpp"
#include "maxpool_golden.hpp"
#include "nearest_neighbor_golden.hpp"

// The quantized 2D-JustoUNet: the same four blocks as justounet2d_golden.hpp,
// but with BatchNorm already folded into each conv by tools/quantize_weights.py
// (so there is no batchnorm_golden call and no eps), and with int8 weights.
//
//   block A: conv(IN_CH->BASE_CH)      -> ReLU -> maxpool(2x2)
//   block B: conv(BASE_CH->2*BASE_CH)  -> ReLU -> maxpool(2x2) -> nearest upsample(2x)
//   block C: conv(2*BASE_CH->BASE_CH)  -> ReLU ->                 nearest upsample(2x)
//   block D: conv(BASE_CH->OUT_CH)                                  (output head)
//
// Weights are the quantized integers q with one power-of-two scale per tensor:
// real value = q * 2^-frac (weights/quantized/.../justounetsimple_int8.hpp). The
// scale is applied once per output value, after the sum, so the inner loop
// multiplies by q itself:
//
//   out = (sum of act * q) * weight_scale + bias_q * bias_scale,   scale = 2^-frac
//
// w_t can be int8_t, or float holding the same integers (what the HLS kernel
// uses, so the hardware needs no int-to-float conversion). With float activations
// the scale is an exact multiplication; with fixed-point activations it becomes
// a plain shift by frac.
//
// Layout is the same as everywhere else: weights [out][in][kh][kw], activations
// HWC. H and W must be divisible by 4. A 32 x 32 patch is the geometry the
// model was trained on.

// 2^-frac, for the scale arguments. constexpr: evaluated by the compiler, so
// no conversion is left for the hardware.
constexpr float pow2_neg(int frac) {
    return 1.0f / (float)(1L << frac);
}

// Same-padded KxK convolution (zero padding) on quantized weights. Out-of-range
// taps are skipped instead of copying the input into a padded buffer, so
// nothing larger than the input and output maps is needed.
template<typename act_t, typename w_t, int H, int W, int IN_CH, int OUT_CH, int K>
void conv2d_same_quant_golden(const act_t din[H][W][IN_CH],
                              const w_t weight[OUT_CH][IN_CH][K][K], act_t weight_scale,
                              const w_t bias[OUT_CH], act_t bias_scale,
                              act_t dout[H][W][OUT_CH]) {
    constexpr int PAD = (K - 1) / 2;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int oc = 0; oc < OUT_CH; oc++) {
                act_t sum = 0;
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            int y = i + ki - PAD;
                            int x = j + kj - PAD;
                            if (y < 0 || y >= H || x < 0 || x >= W) continue;
                            sum += din[y][x][ic] * (act_t)weight[oc][ic][ki][kj];
                        }
                    }
                }
                dout[i][j][oc] = sum * weight_scale + (act_t)bias[oc] * bias_scale;
            }
        }
    }
}

template<typename act_t, typename w_t, int H, int W, int IN_CH, int BASE_CH, int OUT_CH, int K>
void justounetsimple_golden(
    const act_t din[H][W][IN_CH],
    const w_t w1[BASE_CH][IN_CH][K][K],       act_t w1_scale, const w_t b1[BASE_CH],       act_t b1_scale,
    const w_t w2[2*BASE_CH][BASE_CH][K][K],   act_t w2_scale, const w_t b2[2*BASE_CH],     act_t b2_scale,
    const w_t w3[BASE_CH][2*BASE_CH][K][K],   act_t w3_scale, const w_t b3[BASE_CH],       act_t b3_scale,
    const w_t w4[OUT_CH][BASE_CH][K][K],      act_t w4_scale, const w_t b4[OUT_CH],        act_t b4_scale,
    act_t dout[H][W][OUT_CH])
{
    static_assert(H % 4 == 0, "H must be divisible by 4 (block A's halving, then block B's)");
    static_assert(W % 4 == 0, "W must be divisible by 4 (block A's halving, then block B's)");

    constexpr int H2 = H / 2, W2 = W / 2;  // after block A's maxpool
    constexpr int H4 = H2 / 2, W4 = W2 / 2;  // after block B's inner maxpool

    // static: keeps the feature maps off the call stack in the native build.
    // ---- block A ----
    static act_t convA[H][W][BASE_CH];
    static act_t reluA[H][W][BASE_CH];
    static act_t poolA[H2][W2][BASE_CH];
    conv2d_same_quant_golden<act_t, w_t, H, W, IN_CH, BASE_CH, K>(din, w1, w1_scale, b1, b1_scale, convA);
    relu_golden<act_t, H, W, BASE_CH>(convA, reluA);
    maxpool_golden<act_t, H, W, BASE_CH, 2>(reluA, poolA);

    // ---- block B ----
    static act_t convB[H2][W2][2*BASE_CH];
    static act_t reluB[H2][W2][2*BASE_CH];
    static act_t poolB[H4][W4][2*BASE_CH];
    static act_t upB[H2][W2][2*BASE_CH];
    conv2d_same_quant_golden<act_t, w_t, H2, W2, BASE_CH, 2*BASE_CH, K>(poolA, w2, w2_scale, b2, b2_scale, convB);
    relu_golden<act_t, H2, W2, 2*BASE_CH>(convB, reluB);
    maxpool_golden<act_t, H2, W2, 2*BASE_CH, 2>(reluB, poolB);
    nearest_neighbor_golden<act_t, H4, W4, 2*BASE_CH, 2>(poolB, upB);

    // ---- block C ----
    static act_t convC[H2][W2][BASE_CH];
    static act_t reluC[H2][W2][BASE_CH];
    static act_t upC[H][W][BASE_CH];
    conv2d_same_quant_golden<act_t, w_t, H2, W2, 2*BASE_CH, BASE_CH, K>(upB, w3, w3_scale, b3, b3_scale, convC);
    relu_golden<act_t, H2, W2, BASE_CH>(convC, reluC);
    nearest_neighbor_golden<act_t, H2, W2, BASE_CH, 2>(reluC, upC);

    // ---- block D: output head, no ReLU ----
    conv2d_same_quant_golden<act_t, w_t, H, W, BASE_CH, OUT_CH, K>(upC, w4, w4_scale, b4, b4_scale, dout);
}
