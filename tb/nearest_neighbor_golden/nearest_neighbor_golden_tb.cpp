// nearest_neighbor_golden_tb.cpp  --  testbench for nearest_neighbor_golden_top, not a synthesis source.
#include "nearest_neighbor_golden.hpp"
#include <cstdio>
#include <random>

static data_t din[NNUP_HIN][NNUP_WIN][NNUP_CH];
static data_t dout[NNUP_HOUT][NNUP_WOUT][NNUP_CH];
static data_t expected[NNUP_HOUT][NNUP_WOUT][NNUP_CH];

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
    fill_uniform(&din[0][0][0], NNUP_HIN*NNUP_WIN*NNUP_CH, rng);


    nearest_neighbor_golden<data_t, NNUP_HIN, NNUP_WIN, NNUP_CH, NNUP_FACTOR>(din, expected);
    nearest_neighbor_golden_top(din, dout);

    data_t *actual_flat = &dout[0][0][0];
    data_t *expected_flat = &expected[0][0][0];

    int errors = 0;
    for (int i = 0; i < NNUP_HOUT*NNUP_WOUT*NNUP_CH; i++)
        if (actual_flat[i] != expected_flat[i]) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n",
                       i, (double)actual_flat[i], (double)expected_flat[i]);
            errors++;
        }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", NNUP_HOUT*NNUP_WOUT*NNUP_CH);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
