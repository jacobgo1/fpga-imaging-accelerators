// waveletcnn_tb.cpp  --  testbench for the quantized WaveletCNN kernel.
//
// The reference is the float waveletcnn_fp32_golden, fed the same weights
// dequantized (q * 2^-frac) and with BatchNorm set to the identity (the real
// BatchNorm is already folded into conv_block.0 and seg_head.0). It is a
// separate implementation from the kernel's own model, so agreement means the
// int8 + shift arithmetic and the BN-free blocks are right. Results differ
// only by float rounding, so the comparison has a small tolerance.
#include "waveletcnn.hpp"
#include "../../src/golden/waveletcnn_golden.hpp"
#include "../../weights/quantized/waveletcnn/waveletcnn_int8.hpp"
#include <cmath>
#include <cstdio>
#include <random>

namespace q = waveletcnn_int8;

constexpr int H = GOLDEN_PATCH_SIZE, W = GOLDEN_PATCH_SIZE, IC = GOLDEN_BANDS, L = GOLDEN_DWT_TAPS;
constexpr int CC = GOLDEN_WAVELET_CHANNELS, MC = GOLDEN_MID_CHANNELS, OC = GOLDEN_CLASSES;

static float din[H][W][IC];
static float dout[H][W][OC];
static float expected[H][W][OC];

static float h0[1][1][L], h1[1][1][L];
static float proj_w[MC][CC][1][1], proj_b[MC];
static float conv_w[MC][MC][3][3], conv_b[MC];
static float head_w[OC][MC][1][1], head_b[OC];

// Identity BatchNorm for the reference: (x - 0) / sqrt(1 + 0) * 1 + 0.
static float bn_one[MC], bn_zero[MC];

static void dequantize(const int8_t *q_values, float *out, int count, float scale)
{
    for (int i = 0; i < count; i++)
        out[i] = (float)q_values[i] * scale;
}

int main()
{
    dequantize(&q::dwt_spectral_h0[0][0][0], &h0[0][0][0], L, std::ldexp(1.0f, -q::dwt_spectral_h0_frac));
    dequantize(&q::dwt_spectral_h1[0][0][0], &h1[0][0][0], L, std::ldexp(1.0f, -q::dwt_spectral_h1_frac));
    dequantize(&q::projection_weight[0][0][0][0], &proj_w[0][0][0][0], MC * CC, std::ldexp(1.0f, -q::projection_weight_frac));
    dequantize(q::projection_bias, proj_b, MC, std::ldexp(1.0f, -q::projection_bias_frac));
    dequantize(&q::conv_block_0_weight[0][0][0][0], &conv_w[0][0][0][0], MC * MC * 3 * 3, std::ldexp(1.0f, -q::conv_block_0_weight_frac));
    dequantize(q::conv_block_0_bias, conv_b, MC, std::ldexp(1.0f, -q::conv_block_0_bias_frac));
    dequantize(&q::seg_head_0_weight[0][0][0][0], &head_w[0][0][0][0], OC * MC, std::ldexp(1.0f, -q::seg_head_0_weight_frac));
    dequantize(q::seg_head_0_bias, head_b, OC, std::ldexp(1.0f, -q::seg_head_0_bias_frac));

    for (int i = 0; i < MC; i++) { bn_one[i] = 1.0f; bn_zero[i] = 0.0f; }

    std::mt19937 rng(1);  // fixed seed: repeatable
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);  // roughly z-scored bands
    float *in_flat = &din[0][0][0];
    for (int i = 0; i < H * W * IC; i++)
        in_flat[i] = dist(rng);

    waveletcnn_fp32_golden<float, H, W, IC, MC, OC, L>(
        din,
        h0, h1,
        proj_w, proj_b,
        conv_w, conv_b, bn_one, bn_zero, bn_zero, bn_one,
        head_w, head_b, bn_one, bn_zero, bn_zero, bn_one,
        0.0f,
        expected);

    waveletcnn(din, dout);

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
