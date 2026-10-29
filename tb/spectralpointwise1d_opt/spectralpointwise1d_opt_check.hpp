// spectralpointwise1d_opt_check.hpp  --  the checks of the optimized
// 1D-SpectralPointwise kernels' testbenches, shared by the ZCU104's, the
// Zynq-7030's and the Zynq-7020's: the same answers on every board. Before
// including it, a testbench includes its kernel's header and defines
// run_kernel(din, dout, height, width, cycles).
//
// 1. Bit for bit against a plain-loop integer reference written here, apart from
//    the kernel's modules: the z-score, the three layers with their shifts, the
//    ReLUs and saturation, the class.
// 2. Close to the float golden network (spectralpointwise1d_golden.hpp, the int8
//    weights with their scales, float activations) on the same pixels.
// The image is 9 x 7 = 63 pixels: every lane gets blocks, the last one short;
// then 1 x 17, where some lanes get a block of one pixel or none at all. The
// dropped raw bands hold 65535: a wrong band selection shows.
#include "../../src/golden/spectralpointwise1d_golden.hpp"
#include "../../weights/quantized/spectralpointwise1d/spectralpointwise1d_int8.hpp"
#include "../../src/hls/justoliunet_opt/justoliunet_opt_formats.hpp"
#include "../../src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt_formats.hpp"
#include <cmath>
#include <cstdio>
#include <random>

namespace q = spectralpointwise1d_int8;
namespace jf = justoliunet_opt_formats;
namespace fmt = spectralpointwise1d_formats;
constexpr int IMG_H = 9, IMG_W = 7, PIX = IMG_H * IMG_W;

static uint16_t raw[PIX][CUBE_BANDS];
static cube_word_t din[PIX * WORDS_PER_PIXEL];
static score_word_t dout[PIX];
static int32_t want[PIX][4];

static int64_t round_shift64(int64_t s, int shift) {
    return shift == 0 ? s : (s + ((int64_t)1 << (shift - 1))) >> shift;
}

// One layer, plainly: sums at fa + fw fraction bits; with relu, ReLU, round down
// to fout, saturate at ACT_MAX; without, the plain sums.
template<int IN, int OUT>
static void ref_layer(const int64_t in[IN], const int8_t w[OUT][IN][1][1], int fw, const int8_t b[OUT], int fb,
                      int fa, bool relu, int fout, int64_t out[OUT]) {
    for (int o = 0; o < OUT; o++) {
        int64_t s = (int64_t)b[o] << (fa + fw - fb);
        for (int i = 0; i < IN; i++)
            s += in[i] * w[o][i][0][0];
        if (relu) {
            const int64_t r = round_shift64(s, fa + fw - fout);
            out[o] = r < 0 ? 0 : r > jf::ACT_MAX ? jf::ACT_MAX : r;
        } else {
            out[o] = s;
        }
    }
}

static void reference_fixed(const uint16_t px[CUBE_BANDS], int32_t out[4]) {
    int64_t x[110], h1[24], h2[12], s[3];
    for (int k = 0; k < 110; k++) {
        const int64_t v = ((int64_t)px[8 + k] * jf::A[k] + jf::B[k]) >> jf::PREP_FRAC;
        x[k] = v > jf::IN_MAX ? jf::IN_MAX : v < -jf::IN_MAX ? -jf::IN_MAX : v;
    }
    ref_layer<110, 24>(x, q::convsp1_weight, q::convsp1_weight_frac, q::convsp1_bias, q::convsp1_bias_frac,
                       jf::IN_FRAC, true, fmt::ACT_FRAC1, h1);
    ref_layer<24, 12>(h1, q::convsp2_weight, q::convsp2_weight_frac, q::convsp2_bias, q::convsp2_bias_frac,
                      fmt::ACT_FRAC1, true, fmt::ACT_FRAC2, h2);
    ref_layer<12, 3>(h2, q::convsp3_weight, q::convsp3_weight_frac, q::convsp3_bias, q::convsp3_bias_frac,
                     fmt::ACT_FRAC2, false, 0, s);
    const int fa = fmt::ACT_FRAC2 + q::convsp3_weight_frac;
    int best = 0;
    for (int c = 0; c < 3; c++) {
        out[c] = (int32_t)(s[c] << (jf::OUT_FRAC - fa));
        if (out[c] > out[best]) best = c;
    }
    out[3] = best;
}

// The float golden network on the float z-scores of the same pixel.
static void reference_float(const uint16_t px[CUBE_BANDS], float out[3]) {
    float x[110];
    for (int k = 0; k < 110; k++) {
        // mean and std back from A and B: A = 2^(IN_FRAC+PREP_FRAC) / std, B = -mean * A + half
        const double a = jf::A[k], b = (double)jf::B[k] - std::ldexp(1.0, jf::PREP_FRAC - 1);
        x[k] = (float)((px[8 + k] + b / a) * a / std::ldexp(1.0, jf::IN_FRAC + jf::PREP_FRAC));
    }
    spectralpointwise1d_golden<float, int8_t, 110, 24, 12, 3>(
        x, q::convsp1_weight, frac_to_scale(q::convsp1_weight_frac), q::convsp1_bias, frac_to_scale(q::convsp1_bias_frac),
        q::convsp2_weight, frac_to_scale(q::convsp2_weight_frac), q::convsp2_bias, frac_to_scale(q::convsp2_bias_frac),
        q::convsp3_weight, frac_to_scale(q::convsp3_weight_frac), q::convsp3_bias, frac_to_scale(q::convsp3_bias_frac), out);
}

static int compare(int height, int width) {
    const int n = height * width;
    for (int i = 0; i < PIX; i++) dout[i] = score_word_t();
    uint32_t cycles = 0;
    run_kernel(din, dout, height, width, cycles);
    int wrong = 0;
    for (int i = 0; i < PIX; i++)
        for (int c = 0; c < 4; c++) {
            // past the image: nothing written
            const int32_t expect = i < n ? want[i][c] : 0;
            if (dout[i][c] != expect) {
                if (wrong < 5)
                    printf("  %d x %d, pixel %d lane %d: kernel %d, reference %d\n", height, width, i, c,
                           (int)dout[i][c], (int)expect);
                wrong++;
            }
        }
    return wrong;
}

static int check_kernel() {
    // Raw counts whose z-scores are normal on the kept bands (mean and std back
    // from the formats' A and B); 65535 on the dropped ones. Bands drawn
    // independently add up in convsp1 as a real spectrum's do not: at sigma 1 its
    // outputs pass the uint8 range (real pixels reach 11 of its 15.9), which tests
    // the saturation, but makes the scores far from float. So every third pixel
    // has sigma 1, the others 0.25 (inside the range, compared with float).
    std::mt19937 rng(5);
    std::normal_distribution<double> z(0.0, 1.0);
    for (int i = 0; i < PIX; i++)
        for (int r = 0; r < CUBE_BANDS; r++) {
            double v = 65535.0;
            if (r >= 8 && r < 118) {
                const int k = r - 8;
                const double a = jf::A[k], b = (double)jf::B[k] - std::ldexp(1.0, jf::PREP_FRAC - 1);
                const double std_ = std::ldexp(1.0, jf::IN_FRAC + jf::PREP_FRAC) / a, mean = -b / a;
                v = mean + z(rng) * std_ * (i % 3 == 0 ? 1.0 : 0.25);
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
            if (i % 3 != 0) worst = std::fmax(worst, std::fabs(s - f[c]));
            lo = std::fmin(lo, f[c]); hi = std::fmax(hi, f[c]);
            if (f[c] > f[fbest]) fbest = c;
        }
        agree += fbest == want[i][3];
        classes[want[i][3]]++;
    }

    int wrong = compare(IMG_H, IMG_W);
    wrong += compare(1, 17);                  // the first 17 pixels: some lanes get one pixel, or none
    wrong += compare(1, 1);                   // one pixel: lane 0 alone
    // Inside the uint8 range the integer network stays within a few roundings of float.
    const bool close = agree >= PIX * 9 / 10 && hi - lo > 1.0f && worst < 0.5f;
    if (wrong == 0 && close)
        printf("PASS  (%d x %d raw image, 1 x 17 and 1 x 1, every output bit-exact; against float: same class "
               "on %d of %d pixels, worst score difference %.3f (sigma 0.25); classes %d / %d / %d)\n",
               IMG_H, IMG_W, agree, PIX, (double)worst, classes[0], classes[1], classes[2]);
    else
        printf("FAIL  (%d outputs differ from the integer reference; same class as float on %d of %d, "
               "float scores %.2f .. %.2f, worst difference %.3f)\n", wrong, agree, PIX, (double)lo, (double)hi,
               (double)worst);
    return wrong == 0 && close ? 0 : 1;
}
