// justounetsimple_opt_tb.cpp  --  testbench for the optimized kernel.
//
// A small raw cube (40 x 50 pixels x 120 bands, uint16 like a capture): 2 x 2
// patches, the bottom and right ones partly outside the image, so the edge
// repetition is tested too. Two references, patch by patch, cut out the way
// the notebook used to (last row and column repeated):
//  - exact: the kernel's integer arithmetic written as plain loops -- the
//    integer z-score, then the four layers. Integer sums do not depend on the
//    order they are added in, so the kernel must match it bit for bit.
//  - golden: the float kernel (src/hls/justounetsimple/) on the float z-score.
//    The difference is the fixed-point rounding; it must stay small.
// The dropped bands (raw 0-7 and 118-119) hold 65535: they must not count.
#include "justounetsimple_opt.hpp"
#include "justounetsimple_prep.hpp"
#include "../../src/hls/justounetsimple/justounetsimple.cpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

constexpr int IMG_H = 40, IMG_W = 50;  // matches the m_axi depth pragmas
constexpr int H = JOPT_H, W = JOPT_W, IC = JOPT_IN_CH, B = JOPT_BASE_CH, OC = JOPT_OUT_CH;
constexpr int RAW = JOPT_RAW_BANDS, PB = JOPT_PORT_BANDS, FIRST = justounetsimple_prep::FIRST_BAND;
constexpr float SCORE_TOLERANCE = 0.02f;   // fixed point against float, in score units
static_assert(H == JUNETS_H && W == JUNETS_W && IC == JUNETS_IN_CH && OC == JUNETS_OUT_CH,
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

static void exact_reference(const int32_t x[H][W][IC], int32_t scores[H][W][OC]) {
    constexpr int A = JOPT_ACT_FRAC;
    const int s1 = JOPT_IN_FRAC + q::conv1_weight_frac, s2 = A + q::conv2_weight_frac;
    const int s3 = A + q::conv3_weight_frac, s4 = A + q::conv4_weight_frac;
    static int64_t c1[H][W][B], c2[H / 2][W / 2][2 * B], c3[H / 2][W / 2][B], c4[H][W][OC];
    static int32_t a[H / 2][W / 2][B], b[H / 2][W / 2][2 * B], c[H][W][B];

    conv<H, W, IC, B>(x, q::conv1_weight, q::conv1_bias, s1, q::conv1_bias_frac, c1);
    for (int y = 0; y < H / 2; y++)                   // ReLU, pool
        for (int xx = 0; xx < W / 2; xx++)
            for (int ch = 0; ch < B; ch++) {
                int32_t m = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        const int32_t v = requant_relu(c1[2 * y + dy][2 * xx + dx][ch], s1 - A);
                        if (v > m) m = v;
                    }
                a[y][xx][ch] = m;
            }
    conv<H / 2, W / 2, B, 2 * B>(a, q::conv2_weight, q::conv2_bias, s2, q::conv2_bias_frac, c2);
    for (int y = 0; y < H / 2; y++)                   // ReLU, pool, upsample
        for (int xx = 0; xx < W / 2; xx++)
            for (int ch = 0; ch < 2 * B; ch++) {
                int32_t m = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        const int32_t v = requant_relu(c2[(y & ~1) + dy][(xx & ~1) + dx][ch], s2 - A);
                        if (v > m) m = v;
                    }
                b[y][xx][ch] = m;
            }
    conv<H / 2, W / 2, 2 * B, B>(b, q::conv3_weight, q::conv3_bias, s3, q::conv3_bias_frac, c3);
    for (int y = 0; y < H; y++)                       // ReLU, upsample
        for (int xx = 0; xx < W; xx++)
            for (int ch = 0; ch < B; ch++)
                c[y][xx][ch] = requant_relu(c3[y / 2][xx / 2][ch], s3 - A);
    conv<H, W, B, OC>(c, q::conv4_weight, q::conv4_bias, s4, q::conv4_bias_frac, c4);
    const int shift = s4 - JOPT_OUT_FRAC;             // round to OUT_FRAC
    for (int y = 0; y < H; y++)
        for (int xx = 0; xx < W; xx++)
            for (int o = 0; o < OC; o++)
                scores[y][xx][o] = (int32_t)(shift == 0 ? c4[y][xx][o]
                                    : (c4[y][xx][o] + (int64_t(1) << (shift - 1))) >> shift);
}

// The largest |sum| an int16 input can make in conv1: 32767 * sum of |q|.
static double worst_conv1_sum() {
    double worst = 0;
    for (int o = 0; o < B; o++) {
        double s = std::fabs(q::conv1_bias[o]) * std::ldexp(1.0, JOPT_IN_FRAC + q::conv1_weight_frac - q::conv1_bias_frac);
        for (int i = 0; i < IC; i++)
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    s += 32767.0 * std::fabs(q::conv1_weight[o][i][ki][kj]);
        if (s > worst) worst = s;
    }
    return worst;
}

// ---------------------------------------------------------------- test
static uint16_t cube[IMG_H][IMG_W][RAW];
static jopt_raw_t din[IMG_H * IMG_W * RAW / PB];   // the same cube, as the kernel's words
static int32_t dout[IMG_H][IMG_W][OC];
static int32_t exact[IMG_H][IMG_W][OC];
static float golden[IMG_H][IMG_W][OC];

int main()
{
    if (worst_conv1_sum() >= 2147483647.0) {
        printf("FAIL  conv1 can overflow int32 (worst sum %g)\n", worst_conv1_sum());
        return 1;
    }

    // Raw values whose z-score is uniform in -3 .. 3, roughly like a capture.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            for (int r = 0; r < RAW; r++) {
                long v = 65535;                               // dropped band: must not count
                if (r >= FIRST && r < FIRST + IC) {
                    const int k = r - FIRST;
                    v = std::lround(prep::MEAN[k] + dist(rng) / prep::INV_STD[k]);
                    v = v < 0 ? 0 : v > 65535 ? 65535 : v;
                }
                cube[y][x][r] = (uint16_t)v;
                din[(y * IMG_W + x) * (RAW / PB) + r / PB][r % PB] = (uint16_t)v;
            }

    // The references, patch by patch.
    static int32_t xq[H][W][IC];
    static float xf[H][W][IC];
    static int32_t ps[H][W][OC];
    static float pf[H][W][OC];
    for (int y0 = 0; y0 < IMG_H; y0 += H)
        for (int x0 = 0; x0 < IMG_W; x0 += W) {
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    const int yy = y0 + y < IMG_H ? y0 + y : IMG_H - 1;
                    const int xx = x0 + x < IMG_W ? x0 + x : IMG_W - 1;
                    for (int k = 0; k < IC; k++) {
                        const uint16_t dn = cube[yy][xx][FIRST + k];
                        const int64_t s = ((int64_t)dn * prep::A[k] + prep::B[k]) >> prep::FRAC;
                        xq[y][x][k] = (int32_t)(s > 32767 ? 32767 : s < -32768 ? -32768 : s);
                        xf[y][x][k] = ((float)dn - prep::MEAN[k]) * prep::INV_STD[k];
                    }
                }
            exact_reference(xq, ps);
            justounetsimple(xf, pf);
            for (int y = 0; y < H && y0 + y < IMG_H; y++)
                for (int x = 0; x < W && x0 + x < IMG_W; x++)
                    for (int o = 0; o < OC; o++) {
                        exact[y0 + y][x0 + x][o] = ps[y][x][o];
                        golden[y0 + y][x0 + x][o] = pf[y][x][o];
                    }
        }

    justounetsimple_opt(din, din, &dout[0][0][0], IMG_H, IMG_W);

    int errors = 0, far = 0;
    float worst = 0;
    const int32_t *a = &dout[0][0][0], *e = &exact[0][0][0];
    const float *g = &golden[0][0][0];
    constexpr int count = IMG_H * IMG_W * OC;
    const float out_scale = std::ldexp(1.0f, -JOPT_OUT_FRAC);
    for (int i = 0; i < count; i++) {
        if (a[i] != e[i]) {
            if (errors < 5) printf("mismatch at [%d]: got %d, exact reference %d\n", i, a[i], e[i]);
            errors++;
        }
        const float diff = std::fabs(a[i] * out_scale - g[i]);
        if (diff > worst) worst = diff;
        if (diff > SCORE_TOLERANCE) far++;
    }
    if (errors == 0 && far == 0)
        printf("PASS  (%d x %d raw image, %d outputs bit-exact; against float: worst difference %g)\n",
               IMG_H, IMG_W, count, (double)worst);
    else
        printf("FAIL  (%d mismatches with the exact reference, %d outputs further than %g from float)\n",
               errors, far, (double)SCORE_TOLERANCE);
    return errors == 0 && far == 0 ? 0 : 1;
}
