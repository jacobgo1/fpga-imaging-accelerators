// justounetsimple_tb.cpp  --  testbench for the quantized kernel.
//
// The reference is the existing float justounet2d_golden, fed the same weights
// dequantized (q * 2^-frac) and with BatchNorm set to the identity (the real
// BatchNorm is already folded into the weights). It is a separate
// implementation from the kernel's own model, so agreement means the
// int8 + shift arithmetic and the BN-free blocks are right. Results differ
// only by float rounding, so the comparison has a small tolerance.
#include "justounetsimple.hpp"
#include "../../src/golden/justounet2d_golden.hpp"
#include "../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"
#include <cmath>
#include <cstdio>
#include <random>

namespace q = justounetsimple_int8;

constexpr int H = JUNETS_H, W = JUNETS_W, IC = JUNETS_IN_CH;
constexpr int B = JUNETS_BASE_CH, OC = JUNETS_OUT_CH, K = JUNETS_K;

static float din[H][W][IC];
static float dout[H][W][OC];
static float expected[H][W][OC];

static float w1[B][IC][K][K], b1[B];
static float w2[2*B][B][K][K], b2[2*B];
static float w3[B][2*B][K][K], b3[B];
static float w4[OC][B][K][K], b4[OC];

// Identity BatchNorm for the reference: (x - 0) / sqrt(1 + 0) * 1 + 0.
static float bn_one[2*B], bn_zero[2*B];

static void dequantize(const int8_t *q_values, float *out, int count, float scale)
{
    for (int i = 0; i < count; i++)
        out[i] = (float)q_values[i] * scale;
}

int main()
{
    dequantize(&q::conv1_weight[0][0][0][0], &w1[0][0][0][0], B*IC*K*K, std::ldexp(1.0f, -q::conv1_weight_frac));
    dequantize(q::conv1_bias, b1, B, std::ldexp(1.0f, -q::conv1_bias_frac));
    dequantize(&q::conv2_weight[0][0][0][0], &w2[0][0][0][0], 2*B*B*K*K, std::ldexp(1.0f, -q::conv2_weight_frac));
    dequantize(q::conv2_bias, b2, 2*B, std::ldexp(1.0f, -q::conv2_bias_frac));
    dequantize(&q::conv3_weight[0][0][0][0], &w3[0][0][0][0], B*2*B*K*K, std::ldexp(1.0f, -q::conv3_weight_frac));
    dequantize(q::conv3_bias, b3, B, std::ldexp(1.0f, -q::conv3_bias_frac));
    dequantize(&q::conv4_weight[0][0][0][0], &w4[0][0][0][0], OC*B*K*K, std::ldexp(1.0f, -q::conv4_weight_frac));
    dequantize(q::conv4_bias, b4, OC, std::ldexp(1.0f, -q::conv4_bias_frac));

    for (int i = 0; i < 2*B; i++) { bn_one[i] = 1.0f; bn_zero[i] = 0.0f; }

    std::mt19937 rng(1);  // fixed seed: repeatable
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);  // roughly z-scored bands
    float *in_flat = &din[0][0][0];
    for (int i = 0; i < H * W * IC; i++)
        in_flat[i] = dist(rng);

    justounet2d_golden<float, H, W, IC, B, OC, K>(
        din,
        w1, b1, bn_one, bn_zero, bn_zero, bn_one,
        w2, b2, bn_one, bn_zero, bn_zero, bn_one,
        w3, b3, bn_one, bn_zero, bn_zero, bn_one,
        w4, b4, bn_one, bn_zero, bn_zero, bn_one,
        0.0f,
        expected);

    justounetsimple(din, dout);

    const float *actual_flat = &dout[0][0][0];
    const float *expected_flat = &expected[0][0][0];
    int errors = 0;
    float lo = expected_flat[0], hi = expected_flat[0];
    constexpr int out_count = H * W * OC;
    for (int i = 0; i < out_count; i++) {
        float e = expected_flat[i], a = actual_flat[i];
        if (e < lo) lo = e;
        if (e > hi) hi = e;
        if (std::fabs(a - e) > 1e-3f * (1.0f + std::fabs(e))) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n", i, (double)a, (double)e);
            errors++;
        }
    }
    // A reference that is all one value would pass for the wrong reasons.
    if (!(hi - lo > 1e-3f)) {
        printf("FAIL  (reference output is constant: %g)\n", (double)lo);
        return 1;
    }

    if (errors == 0)
        printf("PASS  (%d outputs checked, reference range %g .. %g)\n",
               out_count, (double)lo, (double)hi);
    else
        printf("FAIL  (%d mismatches)\n", errors);
    return errors == 0 ? 0 : 1;
}
