// spectralpointwise1d_tb.cpp  --  testbench for the quantized 1D-SpectralPointwise kernel.
//
// 1. The kernel against the float network as trained, spectralpointwise1d_bn_golden
//    (built from the plain modules), fed the same weights dequantized (q * 2^-frac)
//    and an identity BatchNorm (the real one is already folded into convsp1). A
//    separate implementation from the kernel's own quantized network, so agreement
//    means the int8 + shift arithmetic, the ReLUs and the weight layout are right.
//    Results differ only by float rounding: a small tolerance.
// 2. The BatchNorm itself: the float network with a BatchNorm after convsp1 must
//    equal the same network with that BatchNorm folded into convsp1, which is what
//    tools/quantize_weights.py does to the trained model and why the kernel has no
//    BatchNorm of its own (as justoliunet_bn_tb.cpp).
#include "spectralpointwise1d.hpp"
#include "../../weights/quantized/spectralpointwise1d/spectralpointwise1d_int8.hpp"
#include "../../src/golden/spectralpointwise1d_bn_golden.hpp"
#include <cmath>
#include <cstdio>
#include <random>

namespace q = spectralpointwise1d_int8;

constexpr int PIXELS = 32;   // the kernel's cosim depth is 32 pixels
constexpr int L = SPW_BANDS, C1 = SPW_CH1, C2 = SPW_CH2, C = SPW_CLASSES;

static float w1[C1][L], b1[C1];
static float w2[C2][C1], b2[C2];
static float w3[C][C2], b3[C];
// Identity BatchNorm for the reference: (x - 0) / sqrt(1 + 0) * 1 + 0.
static float bn_one[C1], bn_zero[C1];

static float din[PIXELS][L];
static float dout[PIXELS][C];
static float expected[PIXELS][C];

static void dequantize(const int8_t *q_values, float *out, int count, int frac)
{
    for (int i = 0; i < count; i++)
        out[i] = (float)q_values[i] * std::ldexp(1.0f, -frac);
}

static void load_reference_weights()
{
    // [out][in][1][1] and [out][in] hold the same values in the same order.
    dequantize(&q::convsp1_weight[0][0][0][0], &w1[0][0], C1 * L, q::convsp1_weight_frac);
    dequantize(q::convsp1_bias, b1, C1, q::convsp1_bias_frac);
    dequantize(&q::convsp2_weight[0][0][0][0], &w2[0][0], C2 * C1, q::convsp2_weight_frac);
    dequantize(q::convsp2_bias, b2, C2, q::convsp2_bias_frac);
    dequantize(&q::convsp3_weight[0][0][0][0], &w3[0][0], C * C2, q::convsp3_weight_frac);
    dequantize(q::convsp3_bias, b3, C, q::convsp3_bias_frac);
    for (int i = 0; i < C1; i++) { bn_one[i] = 1.0f; bn_zero[i] = 0.0f; }
}

// The reference scores of one spectrum, with the given convsp1 and BatchNorm.
static void reference(const float spectrum[L], const float cw1[C1][L], const float cb1[C1],
                      const float g[C1], const float beta[C1], const float mean[C1], const float var[C1],
                      float eps, float out[C])
{
    spectralpointwise1d_bn_golden<float, L, C1, C2, C>(spectrum, cw1, cb1, g, beta, mean, var, eps,
                                                       w2, b2, w3, b3, out);
}

// The kernel against the reference on PIXELS random spectra; 0 when it passes.
static int check_kernel()
{
    load_reference_weights();
    std::mt19937 rng(1);  // fixed seed: repeatable
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);  // roughly z-scored bands
    for (int p = 0; p < PIXELS; p++)
        for (int b = 0; b < L; b++)
            din[p][b] = dist(rng);
    for (int p = 0; p < PIXELS; p++)
        reference(din[p], w1, b1, bn_one, bn_zero, bn_zero, bn_one, 0.0f, expected[p]);

    spectralpointwise1d(&din[0][0], &dout[0][0], PIXELS);

    int errors = 0, classes[C] = {0};
    float lo = expected[0][0], hi = expected[0][0];
    for (int p = 0; p < PIXELS; p++) {
        int best = 0;
        for (int c = 0; c < C; c++) {
            const float e = expected[p][c], a = dout[p][c];
            if (e < lo) lo = e;
            if (e > hi) hi = e;
            if (e > expected[p][best]) best = c;
            if (std::fabs(a - e) > 1e-3f * (1.0f + std::fabs(e))) {
                if (errors < 5)
                    printf("mismatch at pixel %d, class %d: got %g, expected %g\n", p, c, (double)a, (double)e);
                errors++;
            }
        }
        classes[best]++;
    }
    // A reference that is all one value would pass for the wrong reasons.
    if (!(hi - lo > 1e-3f)) {
        printf("FAIL  (reference scores are constant: %g)\n", (double)lo);
        return 1;
    }

    // n is honoured: a start with 5 pixels writes their scores and no more.
    static float few[PIXELS][C];
    for (int p = 0; p < PIXELS; p++)
        for (int c = 0; c < C; c++) few[p][c] = -12345.0f;
    spectralpointwise1d(&din[0][0], &few[0][0], 5);
    for (int p = 0; p < PIXELS; p++)
        for (int c = 0; c < C; c++) {
            const bool written = few[p][c] != -12345.0f;
            if (written != (p < 5) || (written && few[p][c] != dout[p][c])) {
                if (errors < 5) printf("n = 5: pixel %d class %d %s\n", p, c, written ? "wrong or extra" : "missing");
                errors++;
            }
        }

    if (errors == 0)
        printf("PASS  (%d pixels x %d scores, reference range %g .. %g, classes %d / %d / %d)\n",
               PIXELS, C, (double)lo, (double)hi, classes[0], classes[1], classes[2]);
    else
        printf("FAIL  (%d mismatches)\n", errors);
    return errors == 0 ? 0 : 1;
}

// BatchNorm with random parameters, against it folded into convsp1:
// scale = gamma / sqrt(var + eps), w1' = w1 * scale, b1' = (b1 - mean) * scale + beta.
static int check_batchnorm()
{
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> any(-1.0f, 1.0f), positive(0.05f, 2.0f);
    static float fw1[C1][L];
    float g[C1], beta[C1], mean[C1], var[C1], fb1[C1];
    const float eps = 1e-5f;
    for (int c = 0; c < C1; c++) {
        g[c] = positive(rng); beta[c] = any(rng); mean[c] = any(rng); var[c] = positive(rng);
        const float scale = g[c] / std::sqrt(var[c] + eps);
        for (int b = 0; b < L; b++) fw1[c][b] = w1[c][b] * scale;
        fb1[c] = (b1[c] - mean[c]) * scale + beta[c];
    }
    int errors = 0;
    for (int p = 0; p < PIXELS; p++) {
        float with_bn[C], folded[C];
        reference(din[p], w1, b1, g, beta, mean, var, eps, with_bn);
        reference(din[p], fw1, fb1, bn_one, bn_zero, bn_zero, bn_one, 0.0f, folded);
        for (int c = 0; c < C; c++)
            if (std::fabs(with_bn[c] - folded[c]) > 1e-4f * (1.0f + std::fabs(folded[c]))) {
                if (errors < 5)
                    printf("BatchNorm: pixel %d class %d: %g with it, %g folded\n", p, c,
                           (double)with_bn[c], (double)folded[c]);
                errors++;
            }
    }
    printf("%s  (BatchNorm after convsp1 = BatchNorm folded into convsp1, %d pixels)\n",
           errors ? "FAIL" : "PASS", PIXELS);
    return errors == 0 ? 0 : 1;
}

int main()
{
    const int kernel = check_kernel();   // also loads the weights and the spectra
    const int bn = check_batchnorm();
    return kernel || bn;
}
