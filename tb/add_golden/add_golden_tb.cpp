// add_golden_tb.cpp  --  testbench for add_golden_top, not a synthesis source.
#include "add_golden.hpp"
#include <cstdio>
#include <random>

static data_t a[ADD_H][ADD_W][ADD_CH];
static data_t b[ADD_H][ADD_W][ADD_CH];
static data_t dout[ADD_H][ADD_W][ADD_CH];
static data_t expected[ADD_H][ADD_W][ADD_CH];

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
    fill_uniform(&a[0][0][0], ADD_H*ADD_W*ADD_CH, rng);
    fill_uniform(&b[0][0][0], ADD_H*ADD_W*ADD_CH, rng);


    add_golden<data_t, ADD_H, ADD_W, ADD_CH>(a, b, expected);
    add_golden_top(a, b, dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    for (int i = 0; i < ADD_H*ADD_W*ADD_CH; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", ADD_H*ADD_W*ADD_CH);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
