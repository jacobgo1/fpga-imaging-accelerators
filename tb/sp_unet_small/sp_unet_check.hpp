// sp_unet_check.hpp  --  the checks of the SpecPW-UNet testbenches, shared by
// sp_unet_small_tb.cpp, sp_unet_medium_tb.cpp and sp_unet_large_tb.cpp. Before
// including it, a testbench includes its kernel's header and its quantized
// weights as namespace weights, and defines run_kernel(din, dout, n).
//
// The reference is the float network as trained, specpw_unet_golden (built from
// the plain modules: BatchNorm, nearest upsampling, the skip's concat as their
// own steps), fed the same weights dequantized (q * 2^-frac) and identity
// BatchNorms (the real ones are already folded into the convs). A folded conv's
// bias goes into its BatchNorm's bias, since the reference's block convs have
// none. It is a separate implementation from the kernel's quantized network, so
// agreement means the int8 + shift arithmetic, the stem read straight from DDR,
// the fused ReLUs and the patch offsets are right. Results differ only by float
// rounding, so the comparison has a small tolerance.
//
// Two patches in one start, different random inputs: the second one checks
// that the kernel reads and writes each patch at its own place.
#include "../../src/golden/specpw_unet_golden.hpp"
#include <cmath>
#include <cstdio>
#include <random>

constexpr int PATCHES = 2;   // the kernel's cosim depth is 2 patches
constexpr int H = GOLDEN_PATCH_SIZE, W = GOLDEN_PATCH_SIZE, IC = GOLDEN_BANDS, OC = GOLDEN_CLASSES;
typedef specpw_unet_params_golden<float, IC, OC, model_size> params_t;

static params_t params;
static float din[PATCHES][H][W][IC];
static float dout[PATCHES][H][W][OC];
static float expected[PATCHES][H][W][OC];

// Same element count, int8 to float, times 2^-frac.
template<typename Q, typename F>
static void dequantize(const Q &q_values, F &out, int frac)
{
    static_assert(sizeof(F) == sizeof(float) * sizeof(Q), "the reference and the weights differ in shape");
    const int8_t *src = reinterpret_cast<const int8_t *>(&q_values);
    float *dst = reinterpret_cast<float *>(&out);
    for (size_t i = 0; i < sizeof(Q); i++)
        dst[i] = (float)src[i] * std::ldexp(1.0f, -frac);
}

// (x - 0) / sqrt(1 + 0) * 1 + bias: adds the folded bias.
template<int CH>
static void identity_bn(batchnorm_params_golden<float, CH> &bn, const float bias[CH])
{
    for (int c = 0; c < CH; c++) {
        bn.weight[c] = 1.0f; bn.bias[c] = bias[c]; bn.running_mean[c] = 0.0f; bn.running_var[c] = 1.0f;
    }
}

template<int OUT, int IN, typename Q, typename B>
static void load_conv_bn(float (&w)[OUT][IN][3][3], batchnorm_params_golden<float, OUT> &bn,
                         const Q &qw, int w_frac, const B &qb, int b_frac)
{
    float bias[OUT];
    dequantize(qw, w, w_frac);
    dequantize(qb, bias, b_frac);
    identity_bn<OUT>(bn, bias);
}

#define CONV_BN(w, bn, name) load_conv_bn(w, bn, weights::name##_weight, weights::name##_weight_frac, \
                                          weights::name##_bias, weights::name##_bias_frac)
#define BLOCK(p, name) CONV_BN(p.conv1_weight, p.bn1, name##_block_0); CONV_BN(p.conv2_weight, p.bn2, name##_block_3)
#define PLAIN(w, b, name) dequantize(weights::name##_weight, w, weights::name##_weight_frac); \
                          dequantize(weights::name##_bias, b, weights::name##_bias_frac)

static void load_reference_weights()
{
    params_t &p = params;
    float zero[model_size::STEM1] = {};
    PLAIN(p.stem.conv1_weight, p.stem.conv1_bias, convsp1);
    identity_bn<model_size::STEM1>(p.stem.bn1, zero);     // bnsp1 is folded into convsp1's bias
    PLAIN(p.stem.conv2_weight, p.stem.conv2_bias, convsp2);
    PLAIN(p.stem.conv3_weight, p.stem.conv3_bias, convsp3);
    BLOCK(p.double1, double1);
    BLOCK(p.double2, double2);
    CONV_BN(p.up1.conv_weight, p.up1.bn, up1_1);
    BLOCK(p.double3, double3);
    CONV_BN(p.up2.conv_weight, p.up2.bn, up2_1);
    BLOCK(p.double4, double4);
    PLAIN(p.final_weight, p.final_bias, final);
}

static int check_kernel()
{
    load_reference_weights();

    std::mt19937 rng(1);  // fixed seed: repeatable
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);  // roughly z-scored bands
    float *in_flat = &din[0][0][0][0];
    for (int i = 0; i < PATCHES * H * W * IC; i++)
        in_flat[i] = dist(rng);

    for (int p = 0; p < PATCHES; p++)
        specpw_unet_golden<float, H, W, IC, OC, model_size>(din[p], params, 0.0f, expected[p]);

    run_kernel(&din[0][0][0][0], &dout[0][0][0][0], PATCHES);

    const float *actual_flat = &dout[0][0][0][0];
    const float *expected_flat = &expected[0][0][0][0];
    int errors = 0;
    float lo = expected_flat[0], hi = expected_flat[0];
    constexpr int out_count = PATCHES * H * W * OC;
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
        printf("PASS  (%d outputs of %d patches checked, reference range %g .. %g)\n",
               out_count, PATCHES, (double)lo, (double)hi);
    else
        printf("FAIL  (%d mismatches)\n", errors);
    return errors == 0 ? 0 : 1;
}
