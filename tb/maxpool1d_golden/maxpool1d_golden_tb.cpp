// maxpool1d_golden_tb.cpp  --  testbench for maxpool1d_golden_top, not a synthesis source.
#include "maxpool1d_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[MP1D_H][MP1D_W][MP1D_CH];
static data_t dout[MP1D_H][MP1D_OUT_W][MP1D_CH];
static data_t expected[MP1D_H][MP1D_OUT_W][MP1D_CH];

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
    fill_uniform(&din[0][0][0], MP1D_H*MP1D_W*MP1D_CH, rng);


    maxpool1d_golden<data_t, MP1D_H, MP1D_W, MP1D_CH, MP1D_POOL>(din, expected);
    maxpool1d_golden_top(din, dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    for (int i = 0; i < MP1D_H*MP1D_OUT_W*MP1D_CH; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", MP1D_H*MP1D_OUT_W*MP1D_CH);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
