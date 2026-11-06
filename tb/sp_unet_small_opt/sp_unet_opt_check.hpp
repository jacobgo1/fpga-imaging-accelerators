// sp_unet_opt_check.hpp  --  the testbench of the optimized SpecPW-UNet kernels
// (sp_unet_SIZE_opt, sp_unet_SIZE_opt_u8 and the Zynq-7020's): the same checks for each.
// A testbench includes its kernel's header, then its golden kernel's .cpp
// (src/models/SpecPW-UNet/SIZE/sp_unet_SIZE_golden/: the float reference, and its weights as namespace weights,
// held as floats with integer values), defines run_kernel(din, dout, height,
// width) and run_golden(din, dout, n), then includes this for check_kernel(): with no
// arguments the int16 arithmetic (every layer ACTIVATION_FRAC_BITS, up to 32767); a uint8
// kernel calls check_kernel_u8 with its layers' fraction bits (SPECPW_UNET_FRACS_GOLDEN(act8)).
//
// A small raw cube (40 x 50 pixels x 120 bands, uint16 like a capture): 2 x 2
// patches, the bottom and right ones partly outside the image, so the edge
// repetition is tested too. Patch by patch, cut out the way the notebook used
// to (last row and column repeated):
//  - exact: the integer z-score, then specpw_unet_fixed_golden (plain loops,
//    one multiply per product, int64 sums: the arithmetic classify_fixed in
//    tools/specpw_unet_model.py does). Integer sums do not depend on the order
//    they are added in, so the kernel must match it
//    bit for bit, and its class must be the first highest of these scores.
//  - golden: the float kernel on the same int8 input, as floats. The
//    difference is the rounding between the layers; it must stay small.
// The z-scores are spread over -4 .. 4, so some inputs saturate at +-127. The
// dropped bands (raw 0-7 and 118-119) hold 65535: they must not count.
#include "../../src/optimized/zscore_int8.hpp"
#include "specpw_unet_fixed_golden.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int IMG_H = 40, IMG_W = 50;      // matches the m_axi depth pragmas
constexpr int FIRST_BAND = zscore_int8::FIRST_BAND;
constexpr float SCORE_TOLERANCE = 0.02f;   // fixed point against float, in score units
// uint8 activations round to 1/16 in the widest layers, so they stay near float, not as
// near (on a real capture 0.14 at most). z-scores of -4 .. 4 in every band, unlike any
// capture, drive ~1 500 activations past the ranges chosen on real data, where they
// saturate and the scores move by several units: so the uint8 kernels are checked bit for
// bit on those (the saturation exercised), and against float on -1 .. 1 (none saturate).
constexpr float U8_SCORE_TOLERANCE = 0.25f, U8_FLOAT_SPREAD = 1.0f;
constexpr int LANES = 4;                   // a pixel of dout: 3 scores and the class
static_assert(PATCH_SIZE == GOLDEN_PATCH_SIZE && MODEL_BANDS == GOLDEN_BANDS && CLASSES == GOLDEN_CLASSES,
              "the optimized and golden kernels must have the same geometry");

namespace prep = zscore_int8;
typedef specpw_unet_quantized_params_golden<MODEL_BANDS, CLASSES, model_size> qparams_t;

// One layer of the golden kernel's float-held weights, as the int8 integers they are.
template<int O, int I, int K, typename QW, typename QB>
static void to_qconv(quantized_conv_golden<O, I, K> &layer, const QW &w, int w_frac, const QB &b, int b_frac) {
    static_assert(sizeof(QW) == sizeof(float) * sizeof(layer.weight) && sizeof(QB) == sizeof(float) * sizeof(layer.bias),
                  "the weights differ in shape");
    const float *wf = reinterpret_cast<const float *>(&w), *bf = reinterpret_cast<const float *>(&b);
    int8_t *wq = &layer.weight[0][0][0][0];
    for (size_t i = 0; i < sizeof(layer.weight); i++) wq[i] = (int8_t)wf[i];
    for (int i = 0; i < O; i++) layer.bias[i] = (int8_t)bf[i];
    layer.weight_frac = w_frac;
    layer.bias_frac = b_frac;
}
#define TO_QCONV(dst, layer) to_qconv(dst, weights::layer##_weight, weights::layer##_weight_frac, \
                                      weights::layer##_bias, weights::layer##_bias_frac)

static void load_qparams(qparams_t &p) {
    TO_QCONV(p.convsp1, convsp1); TO_QCONV(p.convsp2, convsp2); TO_QCONV(p.convsp3, convsp3);
    TO_QCONV(p.double1_conv1, double1_block_0); TO_QCONV(p.double1_conv2, double1_block_3);
    TO_QCONV(p.double2_conv1, double2_block_0); TO_QCONV(p.double2_conv2, double2_block_3);
    TO_QCONV(p.up1, up1_1);
    TO_QCONV(p.double3_conv1, double3_block_0); TO_QCONV(p.double3_conv2, double3_block_3);
    TO_QCONV(p.up2, up2_1);
    TO_QCONV(p.double4_conv1, double4_block_0); TO_QCONV(p.double4_conv2, double4_block_3);
    TO_QCONV(p.final_conv, final);
}

static qparams_t qparams;
static uint16_t cube[IMG_H][IMG_W][CUBE_BANDS];
static cube_word_t din[IMG_H * IMG_W * CUBE_BANDS / BANDS_PER_WORD];  // the same cube, as the kernel's words
static score_word_t dout[IMG_H * IMG_W];
static int32_t exact[IMG_H][IMG_W][CLASSES];
static float golden_scores[IMG_H][IMG_W][CLASSES];

// Raw values whose z-scores are uniform in -spread .. spread; tolerance < 0: bit for bit only.
static int check_kernel(const specpw_unet_fracs_golden &fracs, int32_t act_max, float spread, float tolerance)
{
    load_qparams(qparams);

    // Raw values whose z-score is uniform in -4 .. 4.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-spread, spread);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            for (int r = 0; r < CUBE_BANDS; r++) {
                long v = 65535;                               // dropped band: must not count
                if (r >= FIRST_BAND && r < FIRST_BAND + MODEL_BANDS) {
                    const int k = r - FIRST_BAND;
                    v = std::lround(prep::MEAN[k] + dist(rng) / prep::INV_STD[k]);
                    v = v < 0 ? 0 : v > 65535 ? 65535 : v;
                }
                cube[y][x][r] = (uint16_t)v;
                din[(y * IMG_W + x) * (CUBE_BANDS / BANDS_PER_WORD) + r / BANDS_PER_WORD][r % BANDS_PER_WORD] = (uint16_t)v;
            }

    // The references, patch by patch.
    static int32_t xq[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS], ps[PATCH_SIZE][PATCH_SIZE][CLASSES];
    static float xf[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS], pf[PATCH_SIZE][PATCH_SIZE][CLASSES];
    for (int y0 = 0; y0 < IMG_H; y0 += PATCH_SIZE)
        for (int x0 = 0; x0 < IMG_W; x0 += PATCH_SIZE) {
            for (int y = 0; y < PATCH_SIZE; y++)
                for (int x = 0; x < PATCH_SIZE; x++) {
                    const int yy = y0 + y < IMG_H ? y0 + y : IMG_H - 1;
                    const int xx = x0 + x < IMG_W ? x0 + x : IMG_W - 1;
                    for (int k = 0; k < MODEL_BANDS; k++) {
                        const int64_t s = ((int64_t)cube[yy][xx][FIRST_BAND + k] * prep::A[k] + prep::B[k]) >> prep::FRAC;
                        xq[y][x][k] = (int32_t)(s > prep::IN_MAX ? prep::IN_MAX : s < -prep::IN_MAX ? -prep::IN_MAX : s);
                        xf[y][x][k] = std::ldexp((float)xq[y][x][k], -ZSCORE_FRAC_BITS);
                    }
                }
            specpw_unet_fixed_golden<PATCH_SIZE, PATCH_SIZE, MODEL_BANDS, CLASSES, model_size>(xq, qparams, ZSCORE_FRAC_BITS, fracs, SCORE_FRAC_BITS, act_max, ps);
            run_golden(&xf[0][0][0], &pf[0][0][0], 1);
            for (int y = 0; y < PATCH_SIZE && y0 + y < IMG_H; y++)
                for (int x = 0; x < PATCH_SIZE && x0 + x < IMG_W; x++)
                    for (int o = 0; o < CLASSES; o++) {
                        exact[y0 + y][x0 + x][o] = ps[y][x][o];
                        golden_scores[y0 + y][x0 + x][o] = pf[y][x][o];
                    }
        }

    run_kernel(din, dout, IMG_H, IMG_W);

    int errors = 0, far = 0, classes_differ = 0;
    float worst = 0;
    const float out_scale = std::ldexp(1.0f, -SCORE_FRAC_BITS);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            const score_word_t &a = dout[y * IMG_W + x];
            const int32_t *e = exact[y][x];
            int best = 0, golden_best = 0;
            for (int o = 1; o < CLASSES; o++) {
                if (e[o] > e[best]) best = o;
                if (golden_scores[y][x][o] > golden_scores[y][x][golden_best]) golden_best = o;
            }
            for (int o = 0; o < LANES; o++) {
                const int32_t want = o < CLASSES ? e[o] : best;
                if (a[o] != want) {
                    if (errors < 5) printf("mismatch at (%d, %d) lane %d: got %d, exact reference %d\n",
                                           y, x, o, a[o], want);
                    errors++;
                }
            }
            for (int o = 0; o < CLASSES; o++) {
                const float diff = std::fabs(a[o] * out_scale - golden_scores[y][x][o]);
                if (diff > worst) worst = diff;
                if (tolerance >= 0 && diff > tolerance) far++;
            }
            classes_differ += best != golden_best;
        }
    constexpr int count = IMG_H * IMG_W * LANES;
    const bool pass = errors == 0 && far == 0;
    if (pass)
        printf("PASS  (%d x %d raw image, z-scores -%g .. %g, %d outputs bit-exact; against float%s: worst "
               "difference %g, %d of %d classes differ)\n", IMG_H, IMG_W, (double)spread, (double)spread, count,
               tolerance >= 0 ? "" : " (not checked)", (double)worst, classes_differ, IMG_H * IMG_W);
    else
        printf("FAIL  (%d mismatches with the exact reference, %d scores further than %g from float, worst %g)\n",
               errors, far, (double)tolerance, (double)worst);
    return pass ? 0 : 1;
}

// The int16 kernels.
inline int check_kernel()
{
    return check_kernel(specpw_unet_same_fracs_golden(ACTIVATION_FRAC_BITS), 32767, 4.0f, SCORE_TOLERANCE);
}

// The uint8 kernels (sp_unet_SIZE_opt_u8): saturated at 255, see U8_SCORE_TOLERANCE.
inline int check_kernel_u8(const specpw_unet_fracs_golden &fracs)
{
    const int saturating = check_kernel(fracs, 255, 4.0f, -1.0f);
    const int near_float = check_kernel(fracs, 255, U8_FLOAT_SPREAD, U8_SCORE_TOLERANCE);
    return saturating || near_float;
}
