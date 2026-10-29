// waveletcnn_opt_check.hpp  --  the testbench of the optimized WaveletCNN
// kernels (waveletcnn_opt and its _zynq7020, _zynq7030 versions): the same
// model, the same answers. A kernel's testbench includes its header, defines
// run_kernel(din, dout, height, width), then includes this for main().
//
// A small raw cube (40 x 50 pixels x 120 bands, uint16 like a capture): 2 x 2
// patches, the bottom and right ones partly outside the image, so the edge
// repetition is tested too. Three checks, patch by patch:
//  - exact: the kernel's integer arithmetic written as plain loops -- the
//    integer z-score, the 1D Daubechies-4 DWT, the 1x1 projection, the 3x3
//    conv_block, and the 1x1 seg_head. Every board version must match it bit
//    for bit, and its class must be the first highest of these scores.
//  - the integer z-score within one unit of the float one (saturated alike).
//  - golden: the float kernel (src/hls/waveletcnn/) on the same int8 input, as
//    floats. The difference is the rounding between the layers; it must stay
//    small.
#include "../../src/optimized/shared/zscore_int8.hpp"
#include "../../src/hls/waveletcnn/waveletcnn.cpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

constexpr int IMG_H = 40, IMG_W = 50;  // matches the m_axi depth pragmas
constexpr int FIRST = zscore_int8::FIRST_BAND;
constexpr float SCORE_TOLERANCE = 0.02f;   // fixed point against float, in score units
constexpr int LANES = 4;                   // a pixel of dout: 3 scores and the class
static_assert(PATCH_SIZE == GOLDEN_PATCH_SIZE && MODEL_BANDS == GOLDEN_BANDS && CLASSES == GOLDEN_CLASSES,
              "the optimized and golden kernels must have the same geometry");

namespace q = waveletcnn_int8;
namespace prep = zscore_int8;

static int64_t rshift(int64_t v, int shift) {
    return shift == 0 ? v : (v + (int64_t(1) << (shift - 1))) >> shift;
}

static int64_t leaky_ref(int64_t v) {
    return v >= 0 ? v : rshift(v * 41, 12);
}

static int32_t clamp16(int64_t r) {
    return r > 32767 ? 32767 : r < -32768 ? -32768 : (int32_t)r;
}

static void exact_reference(const int32_t x[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS], int32_t scores[PATCH_SIZE][PATCH_SIZE][CLASSES]) {
    constexpr int PAD = (2 * (SUBBAND_LENGTH - 1) - MODEL_BANDS + DWT_TAPS) / 2;
    const int dwt_frac = ZSCORE_FRAC_BITS + q::dwt_spectral_h0_frac;
    const int s_proj = dwt_frac + q::projection_weight_frac;
    const int s_conv = ACTIVATION_FRAC_BITS + q::conv_block_0_weight_frac;
    const int s_head = ACTIVATION_FRAC_BITS + q::seg_head_0_weight_frac;

    static int32_t cat[PATCH_SIZE][PATCH_SIZE][WAVELET_CHANNELS];
    static int32_t proj[PATCH_SIZE][PATCH_SIZE][MID_CHANNELS];
    static int32_t act1[PATCH_SIZE][PATCH_SIZE][MID_CHANNELS];

    // 1. dwt_spectral + concatenate [x << h0_frac, lo, hi]
    for (int y = 0; y < PATCH_SIZE; y++)
        for (int xx = 0; xx < PATCH_SIZE; xx++) {
            for (int c = 0; c < MODEL_BANDS; c++)
                cat[y][xx][c] = x[y][xx][c] << q::dwt_spectral_h0_frac;
            for (int m = 0; m < SUBBAND_LENGTH; m++) {
                int64_t lo = 0, hi = 0;
                for (int k = 0; k < DWT_TAPS; k++) {
                    const int idx = 2 * m + k - PAD;
                    if (idx < 0 || idx >= MODEL_BANDS) continue;
                    lo += (int64_t)x[y][xx][idx] * (int64_t)q::dwt_spectral_h0[0][0][k];
                    hi += (int64_t)x[y][xx][idx] * (int64_t)q::dwt_spectral_h1[0][0][k];
                }
                cat[y][xx][MODEL_BANDS + m] = (int32_t)lo;
                cat[y][xx][MODEL_BANDS + SUBBAND_LENGTH + m] = (int32_t)hi;
            }
        }

    // 2. 1x1 projection (226 -> 12)
    for (int y = 0; y < PATCH_SIZE; y++)
        for (int xx = 0; xx < PATCH_SIZE; xx++)
            for (int o = 0; o < MID_CHANNELS; o++) {
                int64_t s = (int64_t)q::projection_bias[o] * (int64_t(1) << (s_proj - q::projection_bias_frac));
                for (int i = 0; i < WAVELET_CHANNELS; i++)
                    s += (int64_t)cat[y][xx][i] * (int64_t)q::projection_weight[o][i][0][0];
                if (s != (int32_t)s) { printf("FAIL  a projection sum does not fit int32\n"); exit(1); }
                proj[y][xx][o] = clamp16(rshift(s, s_proj - ACTIVATION_FRAC_BITS));
            }

    // 3. 3x3 conv_block (12 -> 12) + LeakyReLU(0.01)
    for (int y = 0; y < PATCH_SIZE; y++)
        for (int xx = 0; xx < PATCH_SIZE; xx++)
            for (int o = 0; o < MID_CHANNELS; o++) {
                int64_t s = (int64_t)q::conv_block_0_bias[o] * (int64_t(1) << (s_conv - q::conv_block_0_bias_frac));
                for (int i = 0; i < MID_CHANNELS; i++)
                    for (int ki = 0; ki < 3; ki++)
                        for (int kj = 0; kj < 3; kj++) {
                            const int yy = y + ki - 1, xk = xx + kj - 1;
                            if (yy >= 0 && yy < PATCH_SIZE && xk >= 0 && xk < PATCH_SIZE)
                                s += (int64_t)proj[yy][xk][i] * (int64_t)q::conv_block_0_weight[o][i][ki][kj];
                        }
                if (s != (int32_t)s) { printf("FAIL  a conv_block sum does not fit int32\n"); exit(1); }
                act1[y][xx][o] = clamp16(leaky_ref(rshift(s, s_conv - ACTIVATION_FRAC_BITS)));
            }

    // 4. 1x1 seg_head (12 -> 3) + LeakyReLU(0.01)
    for (int y = 0; y < PATCH_SIZE; y++)
        for (int xx = 0; xx < PATCH_SIZE; xx++)
            for (int o = 0; o < CLASSES; o++) {
                int64_t s = (int64_t)q::seg_head_0_bias[o] * (int64_t(1) << (s_head - q::seg_head_0_bias_frac));
                for (int i = 0; i < MID_CHANNELS; i++)
                    s += (int64_t)act1[y][xx][i] * (int64_t)q::seg_head_0_weight[o][i][0][0];
                if (s != (int32_t)s) { printf("FAIL  a seg_head sum does not fit int32\n"); exit(1); }
                scores[y][xx][o] = (int32_t)leaky_ref(rshift(s, s_head - SCORE_FRAC_BITS));
            }
}

static uint16_t cube[IMG_H][IMG_W][CUBE_BANDS];
static cube_word_t din[IMG_H * IMG_W * CUBE_BANDS / BANDS_PER_WORD];
static score_word_t dout[IMG_H * IMG_W];
static int32_t exact[IMG_H][IMG_W][CLASSES];
static float golden[IMG_H][IMG_W][CLASSES];

int main()
{
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
            for (int r = 0; r < CUBE_BANDS; r++) {
                long v = 65535;
                if (r >= FIRST && r < FIRST + MODEL_BANDS) {
                    const int k = r - FIRST;
                    v = std::lround(prep::MEAN[k] + dist(rng) / prep::INV_STD[k]);
                    v = v < 0 ? 0 : v > 65535 ? 65535 : v;
                }
                cube[y][x][r] = (uint16_t)v;
                din[(y * IMG_W + x) * (CUBE_BANDS / BANDS_PER_WORD) + r / BANDS_PER_WORD][r % BANDS_PER_WORD] = (uint16_t)v;
            }

    int z_far = 0;
    static int32_t xq[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS];
    static float xf[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS];
    static int32_t ps[PATCH_SIZE][PATCH_SIZE][CLASSES];
    static float pf[PATCH_SIZE][PATCH_SIZE][CLASSES];
    for (int y0 = 0; y0 < IMG_H; y0 += PATCH_SIZE)
        for (int x0 = 0; x0 < IMG_W; x0 += PATCH_SIZE) {
            for (int y = 0; y < PATCH_SIZE; y++)
                for (int x = 0; x < PATCH_SIZE; x++) {
                    const int yy = y0 + y < IMG_H ? y0 + y : IMG_H - 1;
                    const int xx = x0 + x < IMG_W ? x0 + x : IMG_W - 1;
                    for (int k = 0; k < MODEL_BANDS; k++) {
                        const uint16_t dn = cube[yy][xx][FIRST + k];
                        const int64_t s = ((int64_t)dn * prep::A[k] + prep::B[k]) >> prep::FRAC;
                        xq[y][x][k] = (int32_t)(s > prep::IN_MAX ? prep::IN_MAX : s < -prep::IN_MAX ? -prep::IN_MAX : s);
                        xf[y][x][k] = std::ldexp((float)xq[y][x][k], -ZSCORE_FRAC_BITS);
                        const float z = ((float)dn - prep::MEAN[k]) * prep::INV_STD[k];
                        const long zr = std::lround(std::ldexp(z, ZSCORE_FRAC_BITS));
                        const long zs = zr > prep::IN_MAX ? prep::IN_MAX : zr < -prep::IN_MAX ? -prep::IN_MAX : zr;
                        if (std::labs(zs - xq[y][x][k]) > 1) z_far++;
                    }
                }
            exact_reference(xq, ps);
            waveletcnn(xf, pf);
            for (int y = 0; y < PATCH_SIZE && y0 + y < IMG_H; y++)
                for (int x = 0; x < PATCH_SIZE && x0 + x < IMG_W; x++)
                    for (int o = 0; o < CLASSES; o++) {
                        exact[y0 + y][x0 + x][o] = ps[y][x][o];
                        golden[y0 + y][x0 + x][o] = pf[y][x][o];
                    }
        }

    run_kernel(din, dout, IMG_H, IMG_W);

    int errors = 0, far = 0;
    float worst = 0;
    const float out_scale = std::ldexp(1.0f, -SCORE_FRAC_BITS);
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++) {
            const score_word_t& a = dout[y * IMG_W + x];
            const int32_t *e = exact[y][x];
            int best = 0;
            for (int o = 1; o < CLASSES; o++)
                if (e[o] > e[best]) best = o;
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
