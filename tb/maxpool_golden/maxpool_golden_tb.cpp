// maxpool_golden_tb.cpp  --  testbench for maxpool_golden_top, not a synthesis source.
#include "maxpool_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[MP_H][MP_W][MP_CH];
static data_t dout[MP_OUT_H][MP_OUT_W][MP_CH];
static data_t expected[MP_OUT_H][MP_OUT_W][MP_CH];

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
    fill_uniform(&din[0][0][0], MP_H*MP_W*MP_CH, rng);


    maxpool_golden<data_t, MP_H, MP_W, MP_CH, MP_POOL>(din, expected);
    maxpool_golden_top(din, dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    for (int i = 0; i < MP_OUT_H*MP_OUT_W*MP_CH; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", MP_OUT_H*MP_OUT_W*MP_CH);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
