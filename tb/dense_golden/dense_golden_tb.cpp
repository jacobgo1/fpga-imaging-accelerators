// dense_golden_tb.cpp  --  testbench for dense_golden_top, not a synthesis source.
#include "dense_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[DENSE_H][DENSE_IN];
static data_t weight[DENSE_OUT][DENSE_IN];
static data_t bias[DENSE_OUT];
static data_t dout[DENSE_H][DENSE_OUT];
static data_t expected[DENSE_H][DENSE_OUT];

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
    fill_uniform(&din[0][0], DENSE_H*DENSE_IN, rng);
    fill_uniform(&weight[0][0], DENSE_OUT*DENSE_IN, rng);
    fill_uniform(&bias[0], DENSE_OUT, rng);


    dense_golden<data_t, DENSE_H, DENSE_IN, DENSE_OUT>(din, weight, bias, expected);
    dense_golden_top(din, weight, bias, dout);

    data_t *actual_flat = &dout[0][0];
    data_t *expected_flat = &expected[0][0];

    int errors = 0;
    for (int i = 0; i < DENSE_H*DENSE_OUT; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", DENSE_H*DENSE_OUT);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
