// justounetsimple_opt_check.hpp  --  the testbench of the optimized kernels
// (justounetsimple_opt and its _zynq7020, _zynq7030 versions): the same model, the
// same answers. A kernel's testbench includes its header, defines
// run_kernel(din, dout, height, width) (how its ports take the cube), then
// includes this for main().
//
// A small raw cube (40 x 50 pixels x 120 bands, uint16 like a capture): 2 x 2
// patches, the bottom and right ones partly outside the image, so the edge
// repetition is tested too. Three checks, patch by patch, cut out the way the
// notebook used to (last row and column repeated):
//  - exact: the kernel's integer arithmetic written as plain loops -- the
//    integer z-score, then the four layers, with one multiply per product.
//    Integer sums do not depend on the order they are added in, so the kernel
//    (two products per multiply in the ZCU104's and the 7020's conv1, one or
//    two input channels at a time in the Zynq-7000s' conv2-4) must match it
//    bit for bit, and its class
//    must be the first highest of these scores.
//  - the integer z-score within one unit of the float one (saturated alike).
//  - golden: the float kernel (src/hls/justounetsimple/) on the same int8
//    input, as floats. The difference is the rounding between the layers; it
//    must stay small.
// The z-scores are spread over -5 .. 5, so a fifth of the inputs saturate at
// +-127, the largest products conv1's packing has to hold. The dropped bands
// (raw 0-7 and 118-119) hold 65535: they must not count.
#include "../../src/hls/justounetsimple_opt/justounetsimple_prep.hpp"
#include "../../src/hls/justounetsimple/justounetsimple.cpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

constexpr int IMG_H = 40, IMG_W = 50;  // matches the m_axi depth pragmas
constexpr int FIRST = justounetsimple_prep::FIRST_BAND;
constexpr float SCORE_TOLERANCE = 0.02f;   // fixed point against float, in score units
constexpr int LANES = 4;                   // a pixel of dout: 3 scores and the class
static_assert(H == JUNETS_H && W == JUNETS_W && IN_CH == JUNETS_IN_CH && OUT_CH == JUNETS_OUT_CH,
              "the optimized and golden kernels must have the same geometry");

// ---------------------------------------------------------------- exact reference
// The golden kernel's translation unit holds the weights as floats with
// integer values (see justounetsimple.cpp); here they are those integers.
namespace q = justounetsimple_int8;
namespace prep = justounetsimple_prep;

static int32_t requant_relu(int64_t v, int shift) {
    if (v <= 0) return 0;
    const int64_t r = (v + (int64_t(1) << (shift - 1))) >> shift;
    return r > 32767 ? 32767 : (int32_t)r;
}

template<int HH, int WW, int CI, int CO>
static void conv(const int32_t in[HH][WW][CI], const float w[CO][CI][3][3], const float b[CO],
                 int sum_frac, int bias_frac, int64_t out[HH][WW][CO]) {
    for (int y = 0; y < HH; y++)
        for (int x = 0; x < WW; x++)
            for (int o = 0; o < CO; o++) {
                int64_t s = (int64_t)b[o] * (int64_t(1) << (sum_frac - bias_frac));
                for (int i = 0; i < CI; i++)
                    for (int ki = 0; ki < 3; ki++)
                        for (int kj = 0; kj < 3; kj++) {
                            const int yy = y + ki - 1, xx = x + kj - 1;
                            if (yy >= 0 && yy < HH && xx >= 0 && xx < WW)
                                s += (int64_t)in[yy][xx][i] * (int64_t)w[o][i][ki][kj];
                        }
                if (s != (int32_t)s) { printf("FAIL  a sum does not fit int32\n"); exit(1); }
                out[y][x][o] = s;
            }
}

static void exact_reference(const int32_t x[H][W][IN_CH], int32_t scores[H][W][OUT_CH]) {
    constexpr int A = ACT_FRAC;
    const int s1 = IN_FRAC + q::conv1_weight_frac, s2 = A + q::conv2_weight_frac;
    const int s3 = A + q::conv3_weight_frac, s4 = A + q::conv4_weight_frac;
    static int64_t c1[H][W][BASE_CH], c2[H / 2][W / 2][2 * BASE_CH], c3[H / 2][W / 2][BASE_CH], c4[H][W][OUT_CH];
    static int32_t a[H / 2][W / 2][BASE_CH], b[H / 2][W / 2][2 * BASE_CH], c[H][W][BASE_CH];

    conv<H, W, IN_CH, BASE_CH>(x, q::conv1_weight, q::conv1_bias, s1, q::conv1_bias_frac, c1);
    for (int y = 0; y < H / 2; y++)                   // ReLU, pool
        for (int xx = 0; xx < W / 2; xx++)
            for (int ch = 0; ch < BASE_CH; ch++) {
                int32_t m = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        const int32_t v = requant_relu(c1[2 * y + dy][2 * xx + dx][ch], s1 - A);
                        if (v > m) m = v;
                    }
                a[y][xx][ch] = m;
            }
    conv<H / 2, W / 2, BASE_CH, 2 * BASE_CH>(a, q::conv2_weight, q::conv2_bias, s2, q::conv2_bias_frac, c2);
    for (int y = 0; y < H / 2; y++)                   // ReLU, pool, upsample
        for (int xx = 0; xx < W / 2; xx++)
            for (int ch = 0; ch < 2 * BASE_CH; ch++) {
                int32_t m = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        const int32_t v = requant_relu(c2[(y & ~1) + dy][(xx & ~1) + dx][ch], s2 - A);
                        if (v > m) m = v;
                    }
                b[y][xx][ch] = m;
            }
    conv<H / 2, W / 2, 2 * BASE_CH, BASE_CH>(b, q::conv3_weight, q::conv3_bias, s3, q::conv3_bias_frac, c3);
    for (int y = 0; y < H; y++)                       // ReLU, upsample
        for (int xx = 0; xx < W; xx++)
            for (int ch = 0; ch < BASE_CH; ch++)
                c[y][xx][ch] = requant_relu(c3[y / 2][xx / 2][ch], s3 - A);
    conv<H, W, BASE_CH, OUT_CH>(c, q::conv4_weight, q::conv4_bias, s4, q::conv4_bias_frac, c4);
    const int shift = s4 - OUT_FRAC;                  // round to OUT_FRAC
    for (int y = 0; y < H; y++)
        for (int xx = 0; xx < W; xx++)
            for (int o = 0; o < OUT_CH; o++)
                scores[y][xx][o] = (int32_t)(shift == 0 ? c4[y][xx][o]
                                    : (c4[y][xx][o] + (int64_t(1) << (shift - 1))) >> shift);
}

// The largest |sum| conv1's int8 input can make: 127 * sum of |q|.
static double worst_conv1_sum() {
    double worst = 0;
    for (int o = 0; o < BASE_CH; o++) {
        double s = std::fabs(q::conv1_bias[o]) * std::ldexp(1.0, IN_FRAC + q::conv1_weight_frac - q::conv1_bias_frac);
        for (int i = 0; i < IN_CH; i++)
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    s += prep::IN_MAX * std::fabs(q::conv1_weight[o][i][ki][kj]);
        if (s > worst) worst = s;
    }
    return worst;
}

// ---------------------------------------------------------------- test
static uint16_t cube[IMG_H][IMG_W][RAW_BANDS];
static jopt_raw_t din[IMG_H * IMG_W * RAW_BANDS / PORT_BANDS];  // the same cube, as the kernel's words
static jopt_out_t dout[IMG_H * IMG_W];
static int32_t exact[IMG_H][IMG_W][OUT_CH];
static float golden[IMG_H][IMG_W][OUT_CH];

int main()
{
    if (worst_conv1_sum() >= 2147483647.0) {
        printf("FAIL  conv1 can overflow int32 (worst sum %g)\n", worst_conv1_sum());
        return 1;
    }

    // Raw values whose z-score is uniform in -5 .. 5.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            for (int r = 0; r < RAW_BANDS; r++) {
                long v = 65535;                               // dropped band: must not count
                if (r >= FIRST && r < FIRST + IN_CH) {
                    const int k = r - FIRST;
                    v = std::lround(prep::MEAN[k] + dist(rng) / prep::INV_STD[k]);
                    v = v < 0 ? 0 : v > 65535 ? 65535 : v;
                }
                cube[y][x][r] = (uint16_t)v;
                din[(y * IMG_W + x) * (RAW_BANDS / PORT_BANDS) + r / PORT_BANDS][r % PORT_BANDS] = (uint16_t)v;
            }

    // The references, patch by patch.
    int z_far = 0;
    static int32_t xq[H][W][IN_CH];
    static float xf[H][W][IN_CH];
    static int32_t ps[H][W][OUT_CH];
    static float pf[H][W][OUT_CH];
    for (int y0 = 0; y0 < IMG_H; y0 += H)
        for (int x0 = 0; x0 < IMG_W; x0 += W) {
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    const int yy = y0 + y < IMG_H ? y0 + y : IMG_H - 1;
                    const int xx = x0 + x < IMG_W ? x0 + x : IMG_W - 1;
                    for (int k = 0; k < IN_CH; k++) {
                        const uint16_t dn = cube[yy][xx][FIRST + k];
                        const int64_t s = ((int64_t)dn * prep::A[k] + prep::B[k]) >> prep::FRAC;
                        xq[y][x][k] = (int32_t)(s > prep::IN_MAX ? prep::IN_MAX : s < -prep::IN_MAX ? -prep::IN_MAX : s);
                        xf[y][x][k] = std::ldexp((float)xq[y][x][k], -IN_FRAC);
                        const float z = ((float)dn - prep::MEAN[k]) * prep::INV_STD[k];
                        const long zr = std::lround(std::ldexp(z, IN_FRAC));
                        const long zs = zr > prep::IN_MAX ? prep::IN_MAX : zr < -prep::IN_MAX ? -prep::IN_MAX : zr;
                        if (std::labs(zs - xq[y][x][k]) > 1) z_far++;
                    }
                }
            exact_reference(xq, ps);
            justounetsimple(xf, pf);
            for (int y = 0; y < H && y0 + y < IMG_H; y++)
                for (int x = 0; x < W && x0 + x < IMG_W; x++)
                    for (int o = 0; o < OUT_CH; o++) {
                        exact[y0 + y][x0 + x][o] = ps[y][x][o];
                        golden[y0 + y][x0 + x][o] = pf[y][x][o];
                    }
        }

    run_kernel(din, dout, IMG_H, IMG_W);

    int errors = 0, far = 0;
    float worst = 0;
    const float out_scale = std::ldexp(1.0f, -OUT_FRAC);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            const jopt_out_t& a = dout[y * IMG_W + x];
            const int32_t *e = exact[y][x];
            int best = 0;
            for (int o = 1; o < OUT_CH; o++)
                if (e[o] > e[best]) best = o;
            for (int o = 0; o < LANES; o++) {
                const int32_t want = o < OUT_CH ? e[o] : best;
                if (a[o] != want) {
                    if (errors < 5) printf("mismatch at (%d, %d) lane %d: got %d, exact reference %d\n",
                                           y, x, o, a[o], want);
                    errors++;
                }
            }
            for (int o = 0; o < OUT_CH; o++) {
                const float diff = std::fabs(a[o] * out_scale - golden[y][x][o]);
                if (diff > worst) worst = diff;
                if (diff > SCORE_TOLERANCE) far++;
            }
        }
    constexpr int count = IMG_H * IMG_W * LANES;
    const bool pass = errors == 0 && far == 0 && z_far == 0;
    if (pass)
        printf("PASS  (%d x %d raw image, %d outputs bit-exact; against float: worst difference %g)\n",
               IMG_H, IMG_W, count, (double)worst);
    else
        printf("FAIL  (%d mismatches with the exact reference, %d scores further than %g from float, "
               "%d z-scores more than one unit from float)\n",
               errors, far, (double)SCORE_TOLERANCE, z_far);
    return pass ? 0 : 1;
}
