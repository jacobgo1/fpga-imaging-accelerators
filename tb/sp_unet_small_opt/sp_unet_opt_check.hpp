// sp_unet_opt_check.hpp  --  the testbench of the optimized SpecPW-UNet kernels
// (sp_unet_small_opt, _medium_opt, _large_opt): the same checks for each size.
// A testbench includes its kernel's header, then its golden kernel's .cpp
// (src/hls/sp_unet_SIZE/: the float reference, and its weights as namespace q,
// held as floats with integer values), defines run_kernel(din, dout, height,
// width) and run_golden(din, dout, n), then includes this for check_kernel().
//
// A small raw cube (40 x 50 pixels x 120 bands, uint16 like a capture): 2 x 2
// patches, the bottom and right ones partly outside the image, so the edge
// repetition is tested too. Patch by patch, cut out the way the notebook used
// to (last row and column repeated):
//  - exact: the integer z-score, then specpw_unet_fixed_golden (plain loops,
//    one multiply per product, int64 sums: the arithmetic classify_fixed in
//    tools/specpw_unet_model.py does). Integer sums do not depend on the order
//    they are added in, and Winograd's are exact, so the kernel must match it
//    bit for bit, and its class must be the first highest of these scores.
//  - golden: the float kernel on the same int8 input, as floats. The
//    difference is the rounding between the layers; it must stay small.
// The z-scores are spread over -4 .. 4, so some inputs saturate at +-127. The
// dropped bands (raw 0-7 and 118-119) hold 65535: they must not count.
#include "../../src/hls/justounetsimple_opt/justounetsimple_prep.hpp"
#include "../../src/golden/specpw_unet_fixed_golden.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int IMG_H = 40, IMG_W = 50;      // matches the m_axi depth pragmas
constexpr int FIRST_BAND = justounetsimple_prep::FIRST_BAND;
constexpr float SCORE_TOLERANCE = 0.02f;   // fixed point against float, in score units
constexpr int LANES = 4;                   // a pixel of dout: 3 scores and the class
static_assert(H == SPU_H && W == SPU_W && IN_CH == SPU_IN_CH && CLASSES == SPU_OUT_CH,
              "the optimized and golden kernels must have the same geometry");

namespace prep = justounetsimple_prep;
typedef specpw_unet_qparams_golden<IN_CH, CLASSES, spu_size> qparams_t;

// One layer of the golden kernel's float-held weights, as the int8 integers they are.
template<int O, int I, int K, typename QW, typename QB>
static void to_qconv(qconv_golden<O, I, K> &layer, const QW &w, int w_frac, const QB &b, int b_frac) {
    static_assert(sizeof(QW) == sizeof(float) * sizeof(layer.w) && sizeof(QB) == sizeof(float) * sizeof(layer.b),
                  "the weights differ in shape");
    const float *wf = reinterpret_cast<const float *>(&w), *bf = reinterpret_cast<const float *>(&b);
    int8_t *wq = &layer.w[0][0][0][0];
    for (size_t i = 0; i < sizeof(layer.w); i++) wq[i] = (int8_t)wf[i];
    for (int i = 0; i < O; i++) layer.b[i] = (int8_t)bf[i];
    layer.w_frac = w_frac;
    layer.b_frac = b_frac;
}
#define TO_QCONV(dst, layer) to_qconv(dst, q::layer##_weight, q::layer##_weight_frac, \
                                      q::layer##_bias, q::layer##_bias_frac)

static void load_qparams(qparams_t &p) {
    TO_QCONV(p.sp1, convsp1); TO_QCONV(p.sp2, convsp2); TO_QCONV(p.sp3, convsp3);
    TO_QCONV(p.d1a, double1_block_0); TO_QCONV(p.d1b, double1_block_3);
    TO_QCONV(p.d2a, double2_block_0); TO_QCONV(p.d2b, double2_block_3);
    TO_QCONV(p.up1, up1_1);
    TO_QCONV(p.d3a, double3_block_0); TO_QCONV(p.d3b, double3_block_3);
    TO_QCONV(p.up2, up2_1);
    TO_QCONV(p.d4a, double4_block_0); TO_QCONV(p.d4b, double4_block_3);
    TO_QCONV(p.fin, final);
}

static qparams_t qparams;
static uint16_t cube[IMG_H][IMG_W][RAW_BANDS];
static spu_raw_t din[IMG_H * IMG_W * RAW_BANDS / PORT_BANDS];  // the same cube, as the kernel's words
static spu_out_t dout[IMG_H * IMG_W];
static int32_t exact[IMG_H][IMG_W][CLASSES];
static float golden[IMG_H][IMG_W][CLASSES];

static int check_kernel()
{
    load_qparams(qparams);

    // Raw values whose z-score is uniform in -4 .. 4.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            for (int r = 0; r < RAW_BANDS; r++) {
                long v = 65535;                               // dropped band: must not count
                if (r >= FIRST_BAND && r < FIRST_BAND + IN_CH) {
                    const int k = r - FIRST_BAND;
                    v = std::lround(prep::MEAN[k] + dist(rng) / prep::INV_STD[k]);
                    v = v < 0 ? 0 : v > 65535 ? 65535 : v;
                }
                cube[y][x][r] = (uint16_t)v;
                din[(y * IMG_W + x) * (RAW_BANDS / PORT_BANDS) + r / PORT_BANDS][r % PORT_BANDS] = (uint16_t)v;
            }

    // The references, patch by patch.
    static int32_t xq[H][W][IN_CH], ps[H][W][CLASSES];
    static float xf[H][W][IN_CH], pf[H][W][CLASSES];
    for (int y0 = 0; y0 < IMG_H; y0 += H)
        for (int x0 = 0; x0 < IMG_W; x0 += W) {
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    const int yy = y0 + y < IMG_H ? y0 + y : IMG_H - 1;
                    const int xx = x0 + x < IMG_W ? x0 + x : IMG_W - 1;
                    for (int k = 0; k < IN_CH; k++) {
                        const int64_t s = ((int64_t)cube[yy][xx][FIRST_BAND + k] * prep::A[k] + prep::B[k]) >> prep::FRAC;
                        xq[y][x][k] = (int32_t)(s > prep::IN_MAX ? prep::IN_MAX : s < -prep::IN_MAX ? -prep::IN_MAX : s);
                        xf[y][x][k] = std::ldexp((float)xq[y][x][k], -IN_FRAC);
                    }
                }
            specpw_unet_fixed_golden<H, W, IN_CH, CLASSES, spu_size>(xq, qparams, IN_FRAC, ACT_FRAC, OUT_FRAC, 32767, ps);
            run_golden(&xf[0][0][0], &pf[0][0][0], 1);
            for (int y = 0; y < H && y0 + y < IMG_H; y++)
                for (int x = 0; x < W && x0 + x < IMG_W; x++)
                    for (int o = 0; o < CLASSES; o++) {
                        exact[y0 + y][x0 + x][o] = ps[y][x][o];
                        golden[y0 + y][x0 + x][o] = pf[y][x][o];
                    }
        }

    run_kernel(din, dout, IMG_H, IMG_W);

    int errors = 0, far = 0, classes_differ = 0;
    float worst = 0;
    const float out_scale = std::ldexp(1.0f, -OUT_FRAC);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            const spu_out_t &a = dout[y * IMG_W + x];
            const int32_t *e = exact[y][x];
            int best = 0, golden_best = 0;
            for (int o = 1; o < CLASSES; o++) {
                if (e[o] > e[best]) best = o;
                if (golden[y][x][o] > golden[y][x][golden_best]) golden_best = o;
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
                const float diff = std::fabs(a[o] * out_scale - golden[y][x][o]);
                if (diff > worst) worst = diff;
                if (diff > SCORE_TOLERANCE) far++;
            }
            classes_differ += best != golden_best;
        }
    constexpr int count = IMG_H * IMG_W * LANES;
    const bool pass = errors == 0 && far == 0;
    if (pass)
        printf("PASS  (%d x %d raw image, %d outputs bit-exact; against float: worst difference %g, "
               "%d of %d classes differ)\n", IMG_H, IMG_W, count, (double)worst, classes_differ, IMG_H * IMG_W);
    else
        printf("FAIL  (%d mismatches with the exact reference, %d scores further than %g from float, worst %g)\n",
               errors, far, (double)SCORE_TOLERANCE, (double)worst);
    return pass ? 0 : 1;
}
