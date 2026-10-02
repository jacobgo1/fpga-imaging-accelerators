// batchnorm_golden_tb.cpp  --  testbench for batchnorm_golden_top, not a synthesis source.
#include "batchnorm_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[BN_H][BN_W][BN_CH];
static data_t weight[BN_CH];
static data_t bias[BN_CH];
static data_t running_mean[BN_CH];
static data_t running_var[BN_CH];
static data_t eps;
static data_t dout[BN_H][BN_W][BN_CH];
static data_t expected[BN_H][BN_W][BN_CH];

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
// eps), so it must never be negative, unlike every other fill here.
static void fill_positive(data_t *data, int count, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    for (int i = 0; i < count; i++)
        data[i] = dist(rng);
}


int main()
{
    std::mt19937 rng(1);  // fixed seed: repeatable
    fill_uniform(&din[0][0][0], BN_H*BN_W*BN_CH, rng);
    fill_uniform(&weight[0], BN_CH, rng);
    fill_uniform(&bias[0], BN_CH, rng);
    fill_uniform(&running_mean[0], BN_CH, rng);
    fill_positive(&running_var[0], BN_CH, rng);
    eps = 1e-5f;  // PyTorch's BatchNorm default

    batchnorm_golden<data_t, BN_H, BN_W, BN_CH>(din, weight, bias, running_mean, running_var, eps, expected);
    batchnorm_golden_top(din, weight, bias, running_mean, running_var, eps, dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    for (int i = 0; i < BN_H*BN_W*BN_CH; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", BN_H*BN_W*BN_CH);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
