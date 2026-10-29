// justoliunet_opt_check.hpp  --  the checks of the optimized 1D kernels'
// testbenches, shared by justoliunet_opt and justoliunet_bn_opt. Before
// including it, a testbench includes its kernel's header and its model's int8
// weights as namespace q and activation formats as namespace fmt, and defines
// run_kernel(din, dout, height, width).
//
// 1. Bit for bit against a plain-loop integer reference written here, apart from
//    the kernel's engines: the z-score, the four conv blocks with their shifts,
//    the pools, the channel-major flatten, the dense layer, the class.
// 2. Close to the float golden network (justoliunet_golden.hpp, the int8 weights
//    with their scales, float activations) on the same pixels.
// The image is 9 x 7 = 63 pixels: three blocks of 16 and a last one of 15, so
// both readers and a short last burst; then 1 x 17, so reader 1 gets a block of
// one pixel. The dropped raw bands hold 65535: a wrong band selection shows.
#include "../../src/golden/justoliunet_golden.hpp"
#include "../../src/hls/justoliunet_opt/justoliunet_opt_formats.hpp"
#include <cmath>
#include <cstdio>
#include <random>

namespace jf = justoliunet_opt_formats;
constexpr int IMG_H = 9, IMG_W = 7, PIX = IMG_H * IMG_W;

static uint16_t raw[PIX][CUBE_BANDS];
static cube_word_t din[PIX * WORDS_PER_PIXEL];
static score_word_t dout[PIX];
static int32_t want[PIX][4];

static int64_t round_shift64(int64_t s, int shift) {
    return shift == 0 ? s : (s + ((int64_t)1 << (shift - 1))) >> shift;
}

// One conv block, plainly: valid conv, sums at fa + fw fraction bits, ReLU,
// round down to fout, saturate at ACT_MAX, max-pool 2 (an odd last one dropped).
template<int L, int IC, int OC>
static void ref_block(const int64_t in[L][IC], const int8_t w[OC][IC][KERNEL_SIZE], int fw, const int8_t b[OC], int fb,
                      int fa, int fout, int64_t out[(L - KERNEL_SIZE + 1) / 2][OC]) {
    constexpr int LC = L - KERNEL_SIZE + 1;
    static int64_t a[LC][OC];
    for (int j = 0; j < LC; j++)
        for (int oc = 0; oc < OC; oc++) {
            int64_t s = (int64_t)b[oc] << (fa + fw - fb);
            for (int ic = 0; ic < IC; ic++)
                for (int k = 0; k < KERNEL_SIZE; k++)
                    s += in[j + k][ic] * w[oc][ic][k];
            int64_t r = round_shift64(s, fa + fw - fout);
            a[j][oc] = r < 0 ? 0 : r > jf::ACT_MAX ? jf::ACT_MAX : r;
        }
    for (int j = 0; j < LC / 2; j++)
        for (int oc = 0; oc < OC; oc++)
            out[j][oc] = a[2 * j][oc] > a[2 * j + 1][oc] ? a[2 * j][oc] : a[2 * j + 1][oc];
}

static void reference_fixed(const uint16_t px[CUBE_BANDS], int32_t out[4]) {
    static int64_t x[110][1], p1[52][6], p2[23][12], p3[9][18], p4[2][24];
    for (int k = 0; k < 110; k++) {
        const int64_t s = ((int64_t)px[8 + k] * jf::A[k] + jf::B[k]) >> jf::PREP_FRAC;
        x[k][0] = s > jf::IN_MAX ? jf::IN_MAX : s < -jf::IN_MAX ? -jf::IN_MAX : s;
    }
    ref_block<110, 1, 6>(x, q::conv1_weight, q::conv1_weight_frac, q::conv1_bias, q::conv1_bias_frac,
                         jf::IN_FRAC, fmt::ACT_FRAC1, p1);
    ref_block<52, 6, 12>(p1, q::conv2_weight, q::conv2_weight_frac, q::conv2_bias, q::conv2_bias_frac,
                         fmt::ACT_FRAC1, fmt::ACT_FRAC2, p2);
    ref_block<23, 12, 18>(p2, q::conv3_weight, q::conv3_weight_frac, q::conv3_bias, q::conv3_bias_frac,
                          fmt::ACT_FRAC2, fmt::ACT_FRAC3, p3);
    ref_block<9, 18, 24>(p3, q::conv4_weight, q::conv4_weight_frac, q::conv4_bias, q::conv4_bias_frac,
                         fmt::ACT_FRAC3, fmt::ACT_FRAC4, p4);
    const int fa = fmt::ACT_FRAC4, fw = q::fc_weight_frac;
    int best = 0;
    for (int c = 0; c < 3; c++) {
        int64_t s = (int64_t)q::fc_bias[c] << (fa + fw - q::fc_bias_frac);
        for (int ch = 0; ch < 24; ch++)
            for (int l = 0; l < 2; l++)
                s += p4[l][ch] * q::fc_weight[c][ch * 2 + l];     // channel-major flatten
        out[c] = (int32_t)(s << (jf::OUT_FRAC - fa - fw));
        if (out[c] > out[best]) best = c;
    }
    out[3] = best;
}

// The float golden network on the float z-scores of the same pixel.
static void reference_float(const uint16_t px[CUBE_BANDS], float out[3]) {
    float x[110][1];
    for (int k = 0; k < 110; k++) {
        // mean and std back from A and B: A = 2^(IN_FRAC+PREP_FRAC) / std, B = -mean * A + half
        const double a = jf::A[k], b = (double)jf::B[k] - std::ldexp(1.0, jf::PREP_FRAC - 1);
        x[k][0] = (float)((px[8 + k] + b / a) * a / std::ldexp(1.0, jf::IN_FRAC + jf::PREP_FRAC));
    }
    justoliunet_golden<float, int8_t, 110, 6, 3, KERNEL_SIZE>(
        x, q::conv1_weight, frac_to_scale(q::conv1_weight_frac), q::conv1_bias, frac_to_scale(q::conv1_bias_frac),
        q::conv2_weight, frac_to_scale(q::conv2_weight_frac), q::conv2_bias, frac_to_scale(q::conv2_bias_frac),
        q::conv3_weight, frac_to_scale(q::conv3_weight_frac), q::conv3_bias, frac_to_scale(q::conv3_bias_frac),
        q::conv4_weight, frac_to_scale(q::conv4_weight_frac), q::conv4_bias, frac_to_scale(q::conv4_bias_frac),
        q::fc_weight, frac_to_scale(q::fc_weight_frac), q::fc_bias, frac_to_scale(q::fc_bias_frac), out);
}

static int compare(int height, int width) {
    const int n = height * width;
    for (int i = 0; i < n; i++) dout[i] = score_word_t();
    uint32_t cycles = 0;
    run_kernel(din, dout, height, width, cycles);
    int wrong = 0;
    for (int i = 0; i < n; i++)
        for (int c = 0; c < 4; c++)
            if (dout[i][c] != want[i][c]) {
                if (wrong < 5)
                    printf("  %d x %d, pixel %d lane %d: kernel %d, reference %d\n", height, width, i, c,
                           (int)dout[i][c], (int)want[i][c]);
                wrong++;
            }
    return wrong;
}

static int check_kernel() {
    // Raw counts whose z-scores are N(0, 1) on the kept bands (mean and std back
    // from the formats' A and B); 65535 on the dropped ones.
    std::mt19937 rng(5);
    std::normal_distribution<double> z(0.0, 1.0);
    for (int i = 0; i < PIX; i++)
        for (int r = 0; r < CUBE_BANDS; r++) {
            double v = 65535.0;
            if (r >= 8 && r < 118) {
                const int k = r - 8;
                const double a = jf::A[k], b = (double)jf::B[k] - std::ldexp(1.0, jf::PREP_FRAC - 1);
                const double std_ = std::ldexp(1.0, jf::IN_FRAC + jf::PREP_FRAC) / a, mean = -b / a;
                v = mean + z(rng) * std_;
            }
            raw[i][r] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : std::lround(v));
        }
    for (int i = 0; i < PIX; i++)
        for (int w = 0; w < WORDS_PER_PIXEL; w++)
            for (int l = 0; l < BANDS_PER_WORD; l++)
                din[i * WORDS_PER_PIXEL + w][l] = raw[i][w * BANDS_PER_WORD + l];

    int classes[3] = {0}, agree = 0;
    float worst = 0, lo = 1e9f, hi = -1e9f;
    for (int i = 0; i < PIX; i++) {
        reference_fixed(raw[i], want[i]);
        float f[3];
        reference_float(raw[i], f);
        int fbest = 0;
        for (int c = 0; c < 3; c++) {
            const float s = std::ldexp((float)want[i][c], -jf::OUT_FRAC);
            worst = std::fmax(worst, std::fabs(s - f[c]));
            lo = std::fmin(lo, f[c]); hi = std::fmax(hi, f[c]);
            if (f[c] > f[fbest]) fbest = c;
        }
        agree += fbest == want[i][3];
        classes[want[i][3]]++;
    }

    int wrong = compare(IMG_H, IMG_W);
    wrong += compare(1, 17);                  // the first 17 pixels: blocks of 16 and 1
    const bool close = agree >= PIX * 9 / 10 && hi - lo > 1.0f;
    if (wrong == 0 && close)
        printf("PASS  (%d x %d raw image and 1 x 17, every output bit-exact; against float: same class on %d of %d "
               "pixels, worst score difference %.3f; classes %d / %d / %d)\n",
               IMG_H, IMG_W, agree, PIX, (double)worst, classes[0], classes[1], classes[2]);
    else
        printf("FAIL  (%d outputs differ from the integer reference; same class as float on %d of %d, "
               "float scores %.2f .. %.2f)\n", wrong, agree, PIX, (double)lo, (double)hi);
    return wrong == 0 && close ? 0 : 1;
}
