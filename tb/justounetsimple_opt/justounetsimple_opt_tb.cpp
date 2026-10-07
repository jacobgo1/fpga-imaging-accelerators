// justounetsimple_opt_tb.cpp  --  testbench for the optimized kernel.
//
// Two references, on the same quantized input:
//  - exact: the kernel's integer arithmetic written as plain loops, patch by
//    patch. Integer sums do not depend on the order they are added in, so the
//    kernel must match it bit for bit.
//  - golden: the float kernel (src/hls/justounetsimple/). The difference is
//    the fixed-point rounding between the layers; it must stay small.
// The optimized kernel gets all patches in one call, in its own input layout
// (two ports, [IN_PAD/P1][H][W][PORT_BANDS] each). The padding bands hold
// large values here, not zeros: their weights must be 0.
#include "justounetsimple_opt.hpp"
#include "../../src/hls/justounetsimple/justounetsimple.cpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

constexpr int N = 2;  // patches; matches the m_axi depth pragmas
constexpr int H = JOPT_H, W = JOPT_W, IC = JOPT_IN_CH, B = JOPT_BASE_CH, OC = JOPT_OUT_CH;
constexpr int P1 = JOPT_P1, IP = JOPT_IN_PAD, PB = JOPT_PORT_BANDS;
constexpr float SCORE_TOLERANCE = 0.02f;   // fixed point against float, in score units
static_assert(H == JUNETS_H && W == JUNETS_W && IC == JUNETS_IN_CH && OC == JUNETS_OUT_CH,
              "the optimized and golden kernels must have the same geometry");
static_assert(IP % P1 == 0 && IP >= IC && P1 == 2 * PB, "IN_PAD must be IN_CH rounded up to P1");

// ---------------------------------------------------------------- exact reference
// The golden kernel's translation unit holds the weights as floats with
// integer values (see justounetsimple.cpp); here they are those integers.
namespace q = justounetsimple_int8;

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
static float patches[N][H][W][IC];       // HWC floats, for the golden kernel
static int32_t xq[N][H][W][IC];          // the same patches quantized
static jopt_in_t din[2][N][IP / P1][H][W];   // the optimized kernel's layout, per port
static int32_t dout[N][H][W][OC];
static int32_t exact[N][H][W][OC];
static float golden[N][H][W][OC];

int main()
{
    if (worst_conv1_sum() >= 2147483647.0) {
        printf("FAIL  conv1 can overflow int32 (worst sum %g)\n", worst_conv1_sum());
        return 1;
    }

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);  // roughly z-scored bands
    const float in_scale = std::ldexp(1.0f, JOPT_IN_FRAC);
    for (int k = 0; k < N; k++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int c = 0; c < IP; c++) {
                    int32_t v = 12345;                        // padding: must not count
                    if (c < IC) {
                        v = (int32_t)std::lrint(dist(rng) * in_scale);
                        xq[k][y][x][c] = v;
                        patches[k][y][x][c] = v / in_scale;   // what the kernel sees
                    }
                    const int p = c % P1;
                    din[p / PB][k][c / P1][y][x][p % PB] = (int16_t)v;
                }

    for (int k = 0; k < N; k++) {
        exact_reference(xq[k], exact[k]);
        justounetsimple(patches[k], golden[k]);
    }

    justounetsimple_opt(&din[0][0][0][0][0], &din[1][0][0][0][0], &dout[0][0][0][0], N);

    int errors = 0, far = 0;
    float worst = 0;
    const int32_t *a = &dout[0][0][0][0], *e = &exact[0][0][0][0];
    const float *g = &golden[0][0][0][0];
    constexpr int count = N * H * W * OC;
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
        printf("PASS  (%d patches, %d outputs bit-exact; against float: worst difference %g)\n",
               N, count, (double)worst);
    else
        printf("FAIL  (%d mismatches with the exact reference, %d outputs further than %g from float)\n",
               errors, far, (double)SCORE_TOLERANCE);
    return errors == 0 && far == 0 ? 0 : 1;
}
