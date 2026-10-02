// conv2d_padded_golden_tb.cpp  --  testbench for conv2d_padded_golden_top, not a synthesis source.
#include "conv2d_padded_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[C2DP_H][C2DP_W][C2DP_IN_CH];
static data_t weight[C2DP_OUT_CH][C2DP_IN_CH][C2DP_K][C2DP_K];
static data_t bias[C2DP_OUT_CH];
static data_t dout[C2DP_OUT_H][C2DP_OUT_W][C2DP_OUT_CH];
static data_t expected[C2DP_OUT_H][C2DP_OUT_W][C2DP_OUT_CH];

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
    fill_uniform(&din[0][0][0], C2DP_H*C2DP_W*C2DP_IN_CH, rng);
    fill_uniform(&weight[0][0][0][0], C2DP_OUT_CH*C2DP_IN_CH*C2DP_K*C2DP_K, rng);
    fill_uniform(&bias[0], C2DP_OUT_CH, rng);


    conv2d_padded_golden<data_t, C2DP_H, C2DP_W, C2DP_IN_CH, C2DP_OUT_CH, C2DP_K, C2DP_PAD>(din, weight, bias, expected);
    conv2d_padded_golden_top(din, weight, bias, dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    for (int i = 0; i < C2DP_OUT_H*C2DP_OUT_W*C2DP_OUT_CH; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", C2DP_OUT_H*C2DP_OUT_W*C2DP_OUT_CH);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
