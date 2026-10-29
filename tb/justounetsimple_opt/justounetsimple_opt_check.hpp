// The checks shared by justounetsimple_opt and its Zynq-7020 and 7030 versions.
// A kernel's testbench defines run_kernel(cube, scores, height, width), then includes this.
//
// A random 40 x 50 raw cube: 2 x 2 patches, the bottom and right ones partly outside
// the image. The kernel must match a plain-loop integer reference bit for bit, and
// stay close to the float golden kernel. A fifth of the z-scores saturate, and the
// dropped bands hold 65535, so both must be handled right.
#include "../../src/optimized/shared/zscore_int8.hpp"
#include "../../src/hls/justounetsimple/justounetsimple.cpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace weights = justounetsimple_int8;   // as floats holding the int8 values
namespace zscore = zscore_int8;

constexpr int IMAGE_HEIGHT = 40, IMAGE_WIDTH = 50;   // matches the m_axi depth pragmas
constexpr int FIRST_MODEL_BAND = zscore::FIRST_BAND;
constexpr int VALUES_PER_PIXEL = 4;                  // 3 scores and the class
constexpr float SCORE_TOLERANCE = 0.02f;             // against the float kernel
constexpr int HALF_PATCH = PATCH_SIZE / 2;

static int32_t relu_to_activation(int64_t sum, int shift) {
    if (sum <= 0) return 0;
    const int64_t rounded = (sum + (int64_t(1) << (shift - 1))) >> shift;
    return rounded > 32767 ? 32767 : (int32_t)rounded;
}

template<int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
static void conv3x3_reference(const int32_t in[HEIGHT][WIDTH][IN_CHANNELS],
                              const float weight[OUT_CHANNELS][IN_CHANNELS][3][3], const float bias[OUT_CHANNELS],
                              int sum_frac_bits, int bias_frac_bits, int64_t out[HEIGHT][WIDTH][OUT_CHANNELS]) {
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int o = 0; o < OUT_CHANNELS; o++) {
                int64_t sum = (int64_t)bias[o] * (int64_t(1) << (sum_frac_bits - bias_frac_bits));
                for (int i = 0; i < IN_CHANNELS; i++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++) {
                            const int r = row + dy, c = col + dx;
                            if (r >= 0 && r < HEIGHT && c >= 0 && c < WIDTH)
                                sum += (int64_t)in[r][c][i] * (int64_t)weight[o][i][dy + 1][dx + 1];
                        }
                if (sum != (int32_t)sum) { printf("FAIL  a sum does not fit int32\n"); exit(1); }
                out[row][col][o] = sum;
            }
}

// The kernel's arithmetic, one patch, as plain loops.
static void integer_reference(const int32_t zscores[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS],
                              int32_t scores[PATCH_SIZE][PATCH_SIZE][CLASSES]) {
    const int sum1 = ZSCORE_FRAC_BITS + weights::conv1_weight_frac;
    const int sum2 = ACTIVATION_FRAC_BITS + weights::conv2_weight_frac;
    const int sum3 = ACTIVATION_FRAC_BITS + weights::conv3_weight_frac;
    const int sum4 = ACTIVATION_FRAC_BITS + weights::conv4_weight_frac;

    static int64_t conv1[PATCH_SIZE][PATCH_SIZE][BASE_CHANNELS];
    static int64_t conv2[HALF_PATCH][HALF_PATCH][2 * BASE_CHANNELS];
    static int64_t conv3[HALF_PATCH][HALF_PATCH][BASE_CHANNELS];
    static int64_t conv4[PATCH_SIZE][PATCH_SIZE][CLASSES];
    static int32_t pooled1[HALF_PATCH][HALF_PATCH][BASE_CHANNELS];
    static int32_t pooled2_upsampled[HALF_PATCH][HALF_PATCH][2 * BASE_CHANNELS];
    static int32_t upsampled3[PATCH_SIZE][PATCH_SIZE][BASE_CHANNELS];

    conv3x3_reference<PATCH_SIZE, PATCH_SIZE, MODEL_BANDS, BASE_CHANNELS>(
        zscores, weights::conv1_weight, weights::conv1_bias, sum1, weights::conv1_bias_frac, conv1);
    for (int row = 0; row < HALF_PATCH; row++)
        for (int col = 0; col < HALF_PATCH; col++)
            for (int ch = 0; ch < BASE_CHANNELS; ch++) {
                int32_t largest = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        const int32_t v = relu_to_activation(conv1[2 * row + dy][2 * col + dx][ch], sum1 - ACTIVATION_FRAC_BITS);
                        if (v > largest) largest = v;
                    }
                pooled1[row][col][ch] = largest;
            }

    // Pooled to 8 x 8, then upsampled back to 16 x 16.
    conv3x3_reference<HALF_PATCH, HALF_PATCH, BASE_CHANNELS, 2 * BASE_CHANNELS>(
        pooled1, weights::conv2_weight, weights::conv2_bias, sum2, weights::conv2_bias_frac, conv2);
    for (int row = 0; row < HALF_PATCH; row++)
        for (int col = 0; col < HALF_PATCH; col++)
            for (int ch = 0; ch < 2 * BASE_CHANNELS; ch++) {
                int32_t largest = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        const int32_t v = relu_to_activation(conv2[(row & ~1) + dy][(col & ~1) + dx][ch], sum2 - ACTIVATION_FRAC_BITS);
                        if (v > largest) largest = v;
                    }
                pooled2_upsampled[row][col][ch] = largest;
            }

    conv3x3_reference<HALF_PATCH, HALF_PATCH, 2 * BASE_CHANNELS, BASE_CHANNELS>(
        pooled2_upsampled, weights::conv3_weight, weights::conv3_bias, sum3, weights::conv3_bias_frac, conv3);
    for (int row = 0; row < PATCH_SIZE; row++)
        for (int col = 0; col < PATCH_SIZE; col++)
            for (int ch = 0; ch < BASE_CHANNELS; ch++)
                upsampled3[row][col][ch] = relu_to_activation(conv3[row / 2][col / 2][ch], sum3 - ACTIVATION_FRAC_BITS);

    conv3x3_reference<PATCH_SIZE, PATCH_SIZE, BASE_CHANNELS, CLASSES>(
        upsampled3, weights::conv4_weight, weights::conv4_bias, sum4, weights::conv4_bias_frac, conv4);
    const int shift = sum4 - SCORE_FRAC_BITS;
    for (int row = 0; row < PATCH_SIZE; row++)
        for (int col = 0; col < PATCH_SIZE; col++)
            for (int cls = 0; cls < CLASSES; cls++) {
                const int64_t sum = conv4[row][col][cls];
                scores[row][col][cls] = (int32_t)(shift == 0 ? sum : (sum + (int64_t(1) << (shift - 1))) >> shift);
            }
}

// The largest sum conv1 can make: every z-score saturated, every product positive.
static double largest_conv1_sum() {
    double largest = 0;
    for (int o = 0; o < BASE_CHANNELS; o++) {
        double sum = std::fabs(weights::conv1_bias[o])
                   * std::ldexp(1.0, ZSCORE_FRAC_BITS + weights::conv1_weight_frac - weights::conv1_bias_frac);
        for (int i = 0; i < MODEL_BANDS; i++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                    sum += zscore::IN_MAX * std::fabs(weights::conv1_weight[o][i][ky][kx]);
        if (sum > largest) largest = sum;
    }
    return largest;
}

static uint16_t cube[IMAGE_HEIGHT][IMAGE_WIDTH][CUBE_BANDS];
static cube_word_t cube_words[IMAGE_HEIGHT * IMAGE_WIDTH * CUBE_BANDS / BANDS_PER_WORD];
static score_word_t kernel_scores[IMAGE_HEIGHT * IMAGE_WIDTH];
static int32_t integer_scores[IMAGE_HEIGHT][IMAGE_WIDTH][CLASSES];
static float float_scores[IMAGE_HEIGHT][IMAGE_WIDTH][CLASSES];

int main()
{
    if (largest_conv1_sum() >= 2147483647.0) {
        printf("FAIL  conv1 can overflow int32 (largest sum %g)\n", largest_conv1_sum());
        return 1;
    }

    // Raw values whose z-scores are spread over -5 .. 5.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> random_zscore(-5.0f, 5.0f);
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        for (int x = 0; x < IMAGE_WIDTH; x++)
            for (int band = 0; band < CUBE_BANDS; band++) {
                long value = 65535;
                const int k = band - FIRST_MODEL_BAND;
                if (k >= 0 && k < MODEL_BANDS) {
                    value = std::lround(zscore::MEAN[k] + random_zscore(rng) / zscore::INV_STD[k]);
                    value = value < 0 ? 0 : value > 65535 ? 65535 : value;
                }
                cube[y][x][band] = (uint16_t)value;
                const int word = (y * IMAGE_WIDTH + x) * (CUBE_BANDS / BANDS_PER_WORD) + band / BANDS_PER_WORD;
                cube_words[word][band % BANDS_PER_WORD] = (uint16_t)value;
            }

    // Both references, patch by patch; the edge patches repeat the last row and column.
    int zscores_off_by_more_than_one = 0;
    static int32_t zscores_int[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS];
    static float zscores_float[PATCH_SIZE][PATCH_SIZE][MODEL_BANDS];
    static int32_t patch_integer_scores[PATCH_SIZE][PATCH_SIZE][CLASSES];
    static float patch_float_scores[PATCH_SIZE][PATCH_SIZE][CLASSES];
    for (int top = 0; top < IMAGE_HEIGHT; top += PATCH_SIZE)
        for (int left = 0; left < IMAGE_WIDTH; left += PATCH_SIZE) {
            for (int row = 0; row < PATCH_SIZE; row++)
                for (int col = 0; col < PATCH_SIZE; col++) {
                    const int y = std::min(top + row, IMAGE_HEIGHT - 1);
                    const int x = std::min(left + col, IMAGE_WIDTH - 1);
                    for (int k = 0; k < MODEL_BANDS; k++) {
                        const uint16_t raw = cube[y][x][FIRST_MODEL_BAND + k];
                        const int64_t z = ((int64_t)raw * zscore::A[k] + zscore::B[k]) >> zscore::FRAC;
                        zscores_int[row][col][k] = (int32_t)std::max<int64_t>(-zscore::IN_MAX, std::min<int64_t>(zscore::IN_MAX, z));
                        zscores_float[row][col][k] = std::ldexp((float)zscores_int[row][col][k], -ZSCORE_FRAC_BITS);

                        const float exact = ((float)raw - zscore::MEAN[k]) * zscore::INV_STD[k];
                        const long rounded = std::lround(std::ldexp(exact, ZSCORE_FRAC_BITS));
                        const long saturated = std::max<long>(-zscore::IN_MAX, std::min<long>(zscore::IN_MAX, rounded));
                        if (std::labs(saturated - zscores_int[row][col][k]) > 1) zscores_off_by_more_than_one++;
                    }
                }
            integer_reference(zscores_int, patch_integer_scores);
            justounetsimple(zscores_float, patch_float_scores);
            for (int row = 0; row < PATCH_SIZE && top + row < IMAGE_HEIGHT; row++)
                for (int col = 0; col < PATCH_SIZE && left + col < IMAGE_WIDTH; col++)
                    for (int cls = 0; cls < CLASSES; cls++) {
                        integer_scores[top + row][left + col][cls] = patch_integer_scores[row][col][cls];
                        float_scores[top + row][left + col][cls] = patch_float_scores[row][col][cls];
                    }
        }

    run_kernel(cube_words, kernel_scores, IMAGE_HEIGHT, IMAGE_WIDTH);

    int mismatches = 0, far_from_float = 0;
    float largest_difference = 0;
    const float score_scale = std::ldexp(1.0f, -SCORE_FRAC_BITS);
    for (int y = 0; y < IMAGE_HEIGHT; y++)
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            const score_word_t& got = kernel_scores[y * IMAGE_WIDTH + x];
            const int32_t *want = integer_scores[y][x];
            int best_class = 0;
            for (int cls = 1; cls < CLASSES; cls++)
                if (want[cls] > want[best_class]) best_class = cls;
            for (int i = 0; i < VALUES_PER_PIXEL; i++) {
                const int32_t expected = i < CLASSES ? want[i] : best_class;
                if (got[i] != expected) {
                    if (mismatches < 5)
                        printf("mismatch at (%d, %d) value %d: got %d, integer reference %d\n", y, x, i, got[i], expected);
                    mismatches++;
                }
            }
            for (int cls = 0; cls < CLASSES; cls++) {
                const float difference = std::fabs(got[cls] * score_scale - float_scores[y][x][cls]);
                largest_difference = std::max(largest_difference, difference);
                if (difference > SCORE_TOLERANCE) far_from_float++;
            }
        }

    const bool pass = mismatches == 0 && far_from_float == 0 && zscores_off_by_more_than_one == 0;
    if (pass)
        printf("PASS  (%d x %d raw image, %d outputs bit-exact; against float: worst difference %g)\n",
               IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT * IMAGE_WIDTH * VALUES_PER_PIXEL, (double)largest_difference);
    else
        printf("FAIL  (%d mismatches with the integer reference, %d scores further than %g from float, "
               "%d z-scores more than one unit from float)\n",
               mismatches, far_from_float, (double)SCORE_TOLERANCE, zscores_off_by_more_than_one);
    return pass ? 0 : 1;
}
