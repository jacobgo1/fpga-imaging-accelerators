// streaming_upsample_tb.cpp  --  testbench for streaming_upsample, not a synthesis source
#include "streaming_upsample.hpp"
#include <cstdio>
#include <random>

static supsample_pixel_t input_image   [supsample::in_height][supsample::in_width];
static supsample_pixel_t output_image  [supsample::out_height][supsample::out_width];
static supsample_pixel_t expected_image[supsample::out_height][supsample::out_width];

// Plain, obviously-correct reference: output[r][c] is just a copy of
// input[r/factor][c/factor], nearest-neighbor style.
static void streaming_upsample_reference()
{
    for (int out_row = 0; out_row < supsample::out_height; out_row++)
        for (int out_col = 0; out_col < supsample::out_width; out_col++)
            for (int channel = 0; channel < supsample::channels; channel++)
                expected_image[out_row][out_col].channel[channel] =
                    input_image[out_row / supsample::factor]
                               [out_col / supsample::factor]
                               .channel[channel];
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable
    std::uniform_int_distribution<int> dist(-100, 100);

    for (int in_row = 0; in_row < supsample::in_height; in_row++)
        for (int in_col = 0; in_col < supsample::in_width; in_col++)
            for (int channel = 0; channel < supsample::channels; channel++)
                input_image[in_row][in_col].channel[channel] = data_t(dist(rng));

    streaming_upsample_reference();
    streaming_upsample(input_image, output_image);

    int errors = 0;
    for (int out_row = 0; out_row < supsample::out_height; out_row++)
        for (int out_col = 0; out_col < supsample::out_width; out_col++)
            for (int channel = 0; channel < supsample::channels; channel++)
                if (output_image[out_row][out_col].channel[channel] !=
                    expected_image[out_row][out_col].channel[channel]) {
                    if (errors < 5)
                        printf("mismatch at [%d][%d] channel %d: got %d, expected %d\n",
                               out_row, out_col, channel,
                               output_image[out_row][out_col].channel[channel],
                               expected_image[out_row][out_col].channel[channel]);
                    errors++;
                }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n",
               supsample::out_height * supsample::out_width * supsample::channels);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
