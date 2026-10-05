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
// Weights are int8 with a power-of-two scale per tensor: real value = q * 2^-frac
// (the *_frac constants of weights/quantized/.../justounetsimple_int8.hpp).
// Activations stay act_t (float). The shift is applied once per output value,
// after the sum, so the inner loop multiplies by the raw integer:
//
//   out = (sum of act * q) >> weight_frac + bias_q >> bias_frac
//
// With integer or fixed-point activations ">> frac" is a plain shift, no
// multiplier. With float activations it is an exact multiplication by 2^-frac,
// which HLS builds as a multiply, not a shift.
//
// Layout is the same as everywhere else: weights [out][in][kh][kw], activations
// HWC. H and W must be divisible by 4. A 32 x 32 patch is the geometry the
// model was trained on.

// x * 2^-frac. Exact for floats; write it as x >> frac for integer types.
template<typename act_t>
static inline act_t shift_right(act_t x, int frac) {
    return x * ((act_t)1 / (act_t)(1 << frac));
}

// Same-padded KxK convolution (zero padding) on quantized weights. Out-of-range
// taps are skipped instead of copying the input into a padded buffer, so
// nothing larger than the input and output maps is needed.
template<typename act_t, typename w_t, int H, int W, int IN_CH, int OUT_CH, int K>
void conv2d_same_quant_golden(const act_t din[H][W][IN_CH],
                              const w_t weight[OUT_CH][IN_CH][K][K], int weight_frac,
                              const w_t bias[OUT_CH], int bias_frac,
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
                dout[i][j][oc] = shift_right(sum, weight_frac) + shift_right((act_t)bias[oc], bias_frac);
            }
        }
    }
}

template<typename act_t, typename w_t, int H, int W, int IN_CH, int BASE_CH, int OUT_CH, int K>
void justounetsimple_golden(
    const act_t din[H][W][IN_CH],
    const w_t w1[BASE_CH][IN_CH][K][K],       int w1_frac, const w_t b1[BASE_CH],       int b1_frac,
    const w_t w2[2*BASE_CH][BASE_CH][K][K],   int w2_frac, const w_t b2[2*BASE_CH],     int b2_frac,
    const w_t w3[BASE_CH][2*BASE_CH][K][K],   int w3_frac, const w_t b3[BASE_CH],       int b3_frac,
    const w_t w4[OUT_CH][BASE_CH][K][K],      int w4_frac, const w_t b4[OUT_CH],        int b4_frac,
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
    conv2d_same_quant_golden<act_t, w_t, H, W, IN_CH, BASE_CH, K>(din, w1, w1_frac, b1, b1_frac, convA);
    relu_golden<act_t, H, W, BASE_CH>(convA, reluA);
    maxpool_golden<act_t, H, W, BASE_CH, 2>(reluA, poolA);

    // ---- block B ----
    static act_t convB[H2][W2][2*BASE_CH];
    static act_t reluB[H2][W2][2*BASE_CH];
    static act_t poolB[H4][W4][2*BASE_CH];
    static act_t upB[H2][W2][2*BASE_CH];
    conv2d_same_quant_golden<act_t, w_t, H2, W2, BASE_CH, 2*BASE_CH, K>(poolA, w2, w2_frac, b2, b2_frac, convB);
    relu_golden<act_t, H2, W2, 2*BASE_CH>(convB, reluB);
    maxpool_golden<act_t, H2, W2, 2*BASE_CH, 2>(reluB, poolB);
    nearest_neighbor_golden<act_t, H4, W4, 2*BASE_CH, 2>(poolB, upB);

    // ---- block C ----
    static act_t convC[H2][W2][BASE_CH];
    static act_t reluC[H2][W2][BASE_CH];
    static act_t upC[H][W][BASE_CH];
    conv2d_same_quant_golden<act_t, w_t, H2, W2, 2*BASE_CH, BASE_CH, K>(upB, w3, w3_frac, b3, b3_frac, convC);
    relu_golden<act_t, H2, W2, BASE_CH>(convC, reluC);
    nearest_neighbor_golden<act_t, H2, W2, BASE_CH, 2>(reluC, upC);

    // ---- block D: output head, no ReLU ----
    conv2d_same_quant_golden<act_t, w_t, H, W, BASE_CH, OUT_CH, K>(upC, w4, w4_frac, b4, b4_frac, dout);
}
