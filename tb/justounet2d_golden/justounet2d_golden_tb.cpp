// justounet2d_golden_tb.cpp  --  testbench for justounet2d_golden_top, not
// a synthesis source. The reference is the same justounet2d_golden<float,...>
// call the kernel wraps (see justounet2d_golden.hpp part 1): this checks
// that wiring concrete geometry through interface pragmas didn't change
// what gets computed, same idea as tb/justoliunet's testbench.
#include "justounet2d_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[JUNET2D_H][JUNET2D_W][JUNET2D_IN_CH];

static data_t w1[JUNET2D_BASE_CH][JUNET2D_IN_CH][JUNET2D_K][JUNET2D_K];
static data_t b1[JUNET2D_BASE_CH];
static data_t bn1_weight[JUNET2D_BASE_CH], bn1_bias[JUNET2D_BASE_CH];
static data_t bn1_mean[JUNET2D_BASE_CH],   bn1_var[JUNET2D_BASE_CH];

static data_t w2[2*JUNET2D_BASE_CH][JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K];
static data_t b2[2*JUNET2D_BASE_CH];
static data_t bn2_weight[2*JUNET2D_BASE_CH], bn2_bias[2*JUNET2D_BASE_CH];
static data_t bn2_mean[2*JUNET2D_BASE_CH],   bn2_var[2*JUNET2D_BASE_CH];

static data_t w3[JUNET2D_BASE_CH][2*JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K];
static data_t b3[JUNET2D_BASE_CH];
static data_t bn3_weight[JUNET2D_BASE_CH], bn3_bias[JUNET2D_BASE_CH];
static data_t bn3_mean[JUNET2D_BASE_CH],   bn3_var[JUNET2D_BASE_CH];

static data_t w4[JUNET2D_OUT_CH][JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K];
static data_t b4[JUNET2D_OUT_CH];
static data_t bn4_weight[JUNET2D_OUT_CH], bn4_bias[JUNET2D_OUT_CH];
static data_t bn4_mean[JUNET2D_OUT_CH],   bn4_var[JUNET2D_OUT_CH];

static data_t eps;

static data_t dout[JUNET2D_H][JUNET2D_W][JUNET2D_OUT_CH];
static data_t expected[JUNET2D_H][JUNET2D_W][JUNET2D_OUT_CH];

// Arrays declared above have no padding between elements, so filling them
// as a flat run of `count` values is exactly filling them element by
// element in row-major order (same trick tb/justoliunet uses).
static void fill_uniform(data_t *data, int count, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; i++)
        data[i] = dist(rng);
}

// running_var is a variance: batchnorm_golden takes sqrt(running_var +
// eps), so every *_var vector below must stay positive, unlike every
// other fill here.
static void fill_positive(data_t *data, int count, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    for (int i = 0; i < count; i++)
        data[i] = dist(rng);
}

int main()
{
    std::mt19937 rng(1);  // fixed seed: repeatable

    fill_uniform(&din[0][0][0], JUNET2D_H * JUNET2D_W * JUNET2D_IN_CH, rng);

    fill_uniform(&w1[0][0][0][0], JUNET2D_BASE_CH * JUNET2D_IN_CH * JUNET2D_K * JUNET2D_K, rng);
    fill_uniform(b1, JUNET2D_BASE_CH, rng);
    fill_uniform(bn1_weight, JUNET2D_BASE_CH, rng);
    fill_uniform(bn1_bias, JUNET2D_BASE_CH, rng);
    fill_uniform(bn1_mean, JUNET2D_BASE_CH, rng);
    fill_positive(bn1_var, JUNET2D_BASE_CH, rng);

    fill_uniform(&w2[0][0][0][0], 2*JUNET2D_BASE_CH * JUNET2D_BASE_CH * JUNET2D_K * JUNET2D_K, rng);
    fill_uniform(b2, 2*JUNET2D_BASE_CH, rng);
    fill_uniform(bn2_weight, 2*JUNET2D_BASE_CH, rng);
    fill_uniform(bn2_bias, 2*JUNET2D_BASE_CH, rng);
    fill_uniform(bn2_mean, 2*JUNET2D_BASE_CH, rng);
    fill_positive(bn2_var, 2*JUNET2D_BASE_CH, rng);

    fill_uniform(&w3[0][0][0][0], JUNET2D_BASE_CH * 2*JUNET2D_BASE_CH * JUNET2D_K * JUNET2D_K, rng);
    fill_uniform(b3, JUNET2D_BASE_CH, rng);
    fill_uniform(bn3_weight, JUNET2D_BASE_CH, rng);
    fill_uniform(bn3_bias, JUNET2D_BASE_CH, rng);
    fill_uniform(bn3_mean, JUNET2D_BASE_CH, rng);
    fill_positive(bn3_var, JUNET2D_BASE_CH, rng);

    fill_uniform(&w4[0][0][0][0], JUNET2D_OUT_CH * JUNET2D_BASE_CH * JUNET2D_K * JUNET2D_K, rng);
    fill_uniform(b4, JUNET2D_OUT_CH, rng);
    fill_uniform(bn4_weight, JUNET2D_OUT_CH, rng);
    fill_uniform(bn4_bias, JUNET2D_OUT_CH, rng);
    fill_uniform(bn4_mean, JUNET2D_OUT_CH, rng);
    fill_positive(bn4_var, JUNET2D_OUT_CH, rng);

    eps = 1e-5f;  // PyTorch's BatchNorm default

    justounet2d_golden<data_t, JUNET2D_H, JUNET2D_W, JUNET2D_IN_CH,
                        JUNET2D_BASE_CH, JUNET2D_OUT_CH, JUNET2D_K>(
        din,
        w1, b1, bn1_weight, bn1_bias, bn1_mean, bn1_var,
        w2, b2, bn2_weight, bn2_bias, bn2_mean, bn2_var,
        w3, b3, bn3_weight, bn3_bias, bn3_mean, bn3_var,
        w4, b4, bn4_weight, bn4_bias, bn4_mean, bn4_var,
        eps,
        expected);

    justounet2d_golden_top(
        din,
        w1, b1, bn1_weight, bn1_bias, bn1_mean, bn1_var,
        w2, b2, bn2_weight, bn2_bias, bn2_mean, bn2_var,
        w3, b3, bn3_weight, bn3_bias, bn3_mean, bn3_var,
        w4, b4, bn4_weight, bn4_bias, bn4_mean, bn4_var,
        eps,
        dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    constexpr int out_count = JUNET2D_H * JUNET2D_W * JUNET2D_OUT_CH;
    for (int i = 0; i < out_count; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", out_count);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
