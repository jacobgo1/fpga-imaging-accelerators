// streaming_maxpool_tb.cpp  --  testbench for streaming_maxpool, not a synthesis source
#include "streaming_maxpool.hpp"
#include <cstdio>
#include <random>

static spool_pixel_t input_image   [spool::in_height][spool::in_width];
static spool_pixel_t output_image  [spool::out_height][spool::out_width];
static spool_pixel_t expected_image[spool::out_height][spool::out_width];

// Plain, obviously-correct reference. No line buffer, no pragmas: just the
// textbook definition, reading input_image at whatever (row, col) it wants.
static void streaming_maxpool_reference()
{
    for (int out_row = 0; out_row < spool::out_height; out_row++)
        for (int out_col = 0; out_col < spool::out_width; out_col++)
            for (int channel = 0; channel < spool::channels; channel++) {
                data_t largest = std::numeric_limits<data_t>::min();
                for (int kernel_row = 0; kernel_row < spool::kernel_height; kernel_row++)
                    for (int kernel_col = 0; kernel_col < spool::kernel_width; kernel_col++) {
                        const data_t candidate =
                            input_image[out_row * spool::stride + kernel_row]
                                       [out_col * spool::stride + kernel_col]
                                       .channel[channel];
                        if (candidate > largest)
                            largest = candidate;
                    }
                expected_image[out_row][out_col].channel[channel] = largest;
            }
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable
    std::uniform_int_distribution<int> dist(-100, 100);

    for (int in_row = 0; in_row < spool::in_height; in_row++)
        for (int in_col = 0; in_col < spool::in_width; in_col++)
            for (int channel = 0; channel < spool::channels; channel++)
                input_image[in_row][in_col].channel[channel] = data_t(dist(rng));

    streaming_maxpool_reference();
    streaming_maxpool(input_image, output_image);

    int errors = 0;
    for (int out_row = 0; out_row < spool::out_height; out_row++)
        for (int out_col = 0; out_col < spool::out_width; out_col++)
            for (int channel = 0; channel < spool::channels; channel++)
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
               spool::out_height * spool::out_width * spool::channels);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
