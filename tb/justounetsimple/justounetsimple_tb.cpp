// Testbench of the golden kernel. The reference is justounet2d_golden, a separate
// implementation, given the same weights as floats and BatchNorm as the identity
// (the real BatchNorm is already folded into the weights).
#include "justounetsimple.hpp"
#include "../../src/golden/justounet2d_golden.hpp"
#include "../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"
#include <cmath>
#include <cstdio>
#include <random>

namespace weights = justounetsimple_int8;

constexpr int SIZE = GOLDEN_PATCH_SIZE, BANDS = GOLDEN_BANDS, BASE = GOLDEN_BASE_CHANNELS;
constexpr int CLASSES = GOLDEN_CLASSES, K = GOLDEN_KERNEL_SIZE;

static float patch[SIZE][SIZE][BANDS];
static float scores[SIZE][SIZE][CLASSES];
static float expected[SIZE][SIZE][CLASSES];

static float conv1_weight[BASE][BANDS][K][K], conv1_bias[BASE];
static float conv2_weight[2 * BASE][BASE][K][K], conv2_bias[2 * BASE];
static float conv3_weight[BASE][2 * BASE][K][K], conv3_bias[BASE];
static float conv4_weight[CLASSES][BASE][K][K], conv4_bias[CLASSES];

static float ones[2 * BASE], zeros[2 * BASE];   // identity BatchNorm

static void dequantize(const int8_t *values, float *out, int count, int frac_bits)
{
    for (int i = 0; i < count; i++)
        out[i] = std::ldexp((float)values[i], -frac_bits);
}

int main()
{
    dequantize(&weights::conv1_weight[0][0][0][0], &conv1_weight[0][0][0][0], BASE * BANDS * K * K, weights::conv1_weight_frac);
    dequantize(&weights::conv2_weight[0][0][0][0], &conv2_weight[0][0][0][0], 2 * BASE * BASE * K * K, weights::conv2_weight_frac);
    dequantize(&weights::conv3_weight[0][0][0][0], &conv3_weight[0][0][0][0], BASE * 2 * BASE * K * K, weights::conv3_weight_frac);
    dequantize(&weights::conv4_weight[0][0][0][0], &conv4_weight[0][0][0][0], CLASSES * BASE * K * K, weights::conv4_weight_frac);
    dequantize(weights::conv1_bias, conv1_bias, BASE, weights::conv1_bias_frac);
    dequantize(weights::conv2_bias, conv2_bias, 2 * BASE, weights::conv2_bias_frac);
    dequantize(weights::conv3_bias, conv3_bias, BASE, weights::conv3_bias_frac);
    dequantize(weights::conv4_bias, conv4_bias, CLASSES, weights::conv4_bias_frac);
    for (int i = 0; i < 2 * BASE; i++) { ones[i] = 1.0f; zeros[i] = 0.0f; }

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> random_zscore(-3.0f, 3.0f);
    float *pixels = &patch[0][0][0];
    for (int i = 0; i < SIZE * SIZE * BANDS; i++)
        pixels[i] = random_zscore(rng);

    justounet2d_golden<float, SIZE, SIZE, BANDS, BASE, CLASSES, K>(
        patch,
        conv1_weight, conv1_bias, ones, zeros, zeros, ones,
        conv2_weight, conv2_bias, ones, zeros, zeros, ones,
        conv3_weight, conv3_bias, ones, zeros, zeros, ones,
        conv4_weight, conv4_bias, ones, zeros, zeros, ones,
        0.0f,
        expected);

    justounetsimple(patch, scores);

    const float *got = &scores[0][0][0];
    const float *want = &expected[0][0][0];
    constexpr int count = SIZE * SIZE * CLASSES;
    int mismatches = 0;
    float lowest = want[0], highest = want[0];
    for (int i = 0; i < count; i++) {
        lowest = std::fmin(lowest, want[i]);
        highest = std::fmax(highest, want[i]);
        if (std::fabs(got[i] - want[i]) > 1e-3f * (1.0f + std::fabs(want[i]))) {
            if (mismatches < 5)
                printf("mismatch at [%d]: got %g, expected %g\n", i, (double)got[i], (double)want[i]);
            mismatches++;
        }
    }
    // A reference that is all one value would pass for the wrong reasons.
    if (!(highest - lowest > 1e-3f)) {
        printf("FAIL  (reference output is constant: %g)\n", (double)lowest);
        return 1;
    }

    if (mismatches == 0)
        printf("PASS  (%d outputs checked, reference range %g .. %g)\n", count, (double)lowest, (double)highest);
    else
        printf("FAIL  (%d mismatches)\n", mismatches);
    return mismatches == 0 ? 0 : 1;
}
