// justoliunet_check.hpp  --  the checks of the justoliunet testbenches, shared
// by justoliunet_tb.cpp and justoliunet_bn_tb.cpp. Before including it, a
// testbench includes its kernel's header and its quantized weights as
// namespace q, and defines run_kernel(din, dout, n).
//
// The reference is the float network as trained, justoliunet_bn_golden (built
// from the plain modules), fed the same weights dequantized (q * 2^-frac) and
// an identity BatchNorm (the real one is already folded into conv1). It is a
// separate implementation from the kernel's own quantized network, so agreement
// means the int8 + shift arithmetic, the valid convs, the pools that drop an
// odd last position and the channel-major flatten are right. Results differ
// only by float rounding, so the comparison has a small tolerance.
#include "../../src/golden/justoliunet_bn_golden.hpp"
#include <cmath>
#include <cstdio>
#include <random>

constexpr int PIXELS = 32;   // the kernel's cosim depth is 32 pixels
constexpr int L = JLN_BANDS, B = JLN_BASE_CH, C = JLN_CLASSES, K = JLN_K;
constexpr int FLAT = 4 * B * justoliunet_flat_len(L, K);
static_assert(FLAT == 48, "110 bands, k = 6, 6 base channels: the checkpoints' fc takes 48");

static float w1[B][1][K], b1[B];
static float w2[2*B][B][K], b2[2*B];
static float w3[3*B][2*B][K], b3[3*B];
static float w4[4*B][3*B][K], b4[4*B];
static float wf[C][FLAT], bf[C];
// Identity BatchNorm for the reference: (x - 0) / sqrt(1 + 0) * 1 + 0.
static float bn_one[B], bn_zero[B];

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
    dequantize(&q::conv1_weight[0][0][0], &w1[0][0][0], B*1*K, q::conv1_weight_frac);
    dequantize(q::conv1_bias, b1, B, q::conv1_bias_frac);
    dequantize(&q::conv2_weight[0][0][0], &w2[0][0][0], 2*B*B*K, q::conv2_weight_frac);
    dequantize(q::conv2_bias, b2, 2*B, q::conv2_bias_frac);
    dequantize(&q::conv3_weight[0][0][0], &w3[0][0][0], 3*B*2*B*K, q::conv3_weight_frac);
    dequantize(q::conv3_bias, b3, 3*B, q::conv3_bias_frac);
    dequantize(&q::conv4_weight[0][0][0], &w4[0][0][0], 4*B*3*B*K, q::conv4_weight_frac);
    dequantize(q::conv4_bias, b4, 4*B, q::conv4_bias_frac);
    dequantize(&q::fc_weight[0][0], &wf[0][0], C*FLAT, q::fc_weight_frac);
    dequantize(q::fc_bias, bf, C, q::fc_bias_frac);
    for (int i = 0; i < B; i++) { bn_one[i] = 1.0f; bn_zero[i] = 0.0f; }
}

// The reference scores of one spectrum, with the given conv1 and BatchNorm.
static void reference(const float spectrum[L], const float cw1[B][1][K], const float cb1[B],
                      const float g[B], const float beta[B], const float mean[B], const float var[B],
                      float eps, float out[C])
{
    float x[L][1];
    for (int b = 0; b < L; b++) x[b][0] = spectrum[b];
    justoliunet_bn_golden<float, L, B, C, K>(x, cw1, cb1, g, beta, mean, var, eps,
                                             w2, b2, w3, b3, w4, b4, wf, bf, out);
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

    run_kernel(&din[0][0], &dout[0][0], PIXELS);

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
    run_kernel(&din[0][0], &few[0][0], 5);
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
