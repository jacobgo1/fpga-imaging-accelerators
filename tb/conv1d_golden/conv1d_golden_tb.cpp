// conv1d_golden_tb.cpp  --  testbench for conv1d_golden_top, not a synthesis source.
#include "conv1d_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[C1D_H][C1D_W][C1D_IN_CH];
static data_t weight[C1D_OUT_CH][C1D_IN_CH][C1D_K];
static data_t bias[C1D_OUT_CH];
static data_t dout[C1D_H][C1D_OUT_W][C1D_OUT_CH];
static data_t expected[C1D_H][C1D_OUT_W][C1D_OUT_CH];

// Arrays declared above have no padding between elements, so filling them
// as a flat run of `count` values is exactly filling them element by
// element in row-major order (same trick tb/justoliunet uses).
static void fill_uniform(data_t *data, int count, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; i++)
        data[i] = dist(rng);
}


int main()
{
    std::mt19937 rng(1);  // fixed seed: repeatable
    fill_uniform(&din[0][0][0], C1D_H*C1D_W*C1D_IN_CH, rng);
    fill_uniform(&weight[0][0][0], C1D_OUT_CH*C1D_IN_CH*C1D_K, rng);
    fill_uniform(&bias[0], C1D_OUT_CH, rng);


    conv1d_golden<data_t, C1D_H, C1D_W, C1D_IN_CH, C1D_OUT_CH, C1D_K>(din, weight, bias, expected);
    conv1d_golden_top(din, weight, bias, dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    for (int i = 0; i < C1D_H*C1D_OUT_W*C1D_OUT_CH; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", C1D_H*C1D_OUT_W*C1D_OUT_CH);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
