// streaming_conv2d_tb.cpp  --  testbench for streaming_conv2d, not a synthesis source
#include "streaming_conv2d.hpp"
#include <cstdio>
#include <random>

static sconv_in_pixel_t  input_image [sconv::in_height][sconv::in_width];
static data_t            kernel_weights[sconv::out_channels][sconv::in_channels]
                                       [sconv::kernel_height][sconv::kernel_width];
static sconv_out_pixel_t output_image  [sconv::out_height][sconv::out_width];
static sconv_out_pixel_t expected_image[sconv::out_height][sconv::out_width];

// Plain, obviously-correct reference. No line buffer, no pragmas: just the
// textbook definition, reading input_image at whatever (row, col) it wants.
static void streaming_conv2d_reference()
{
    for (int out_row = 0; out_row < sconv::out_height; out_row++)
        for (int out_col = 0; out_col < sconv::out_width; out_col++)
            for (int out_channel = 0; out_channel < sconv::out_channels; out_channel++) {
                result_t sum = 0;
                for (int in_channel = 0; in_channel < sconv::in_channels; in_channel++)
                    for (int kernel_row = 0; kernel_row < sconv::kernel_height; kernel_row++)
                        for (int kernel_col = 0; kernel_col < sconv::kernel_width; kernel_col++)
                            sum += result_t(input_image[out_row * sconv::stride + kernel_row]
                                                        [out_col * sconv::stride + kernel_col]
                                                        .channel[in_channel])
                                 * result_t(kernel_weights[out_channel][in_channel]
                                                          [kernel_row][kernel_col]);
                expected_image[out_row][out_col].channel[out_channel] = sum;
            }
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable
    std::uniform_int_distribution<int> dist(-100, 100);

    for (int in_row = 0; in_row < sconv::in_height; in_row++)
        for (int in_col = 0; in_col < sconv::in_width; in_col++)
            for (int in_channel = 0; in_channel < sconv::in_channels; in_channel++)
                input_image[in_row][in_col].channel[in_channel] = data_t(dist(rng));

    for (int out_channel = 0; out_channel < sconv::out_channels; out_channel++)
        for (int in_channel = 0; in_channel < sconv::in_channels; in_channel++)
            for (int kernel_row = 0; kernel_row < sconv::kernel_height; kernel_row++)
                for (int kernel_col = 0; kernel_col < sconv::kernel_width; kernel_col++)
                    kernel_weights[out_channel][in_channel][kernel_row][kernel_col] =
                        data_t(dist(rng));

    streaming_conv2d_reference();
    streaming_conv2d(input_image, kernel_weights, output_image);

    int errors = 0;
    for (int out_row = 0; out_row < sconv::out_height; out_row++)
        for (int out_col = 0; out_col < sconv::out_width; out_col++)
            for (int out_channel = 0; out_channel < sconv::out_channels; out_channel++)
                if (output_image[out_row][out_col].channel[out_channel] !=
                    expected_image[out_row][out_col].channel[out_channel]) {
                    if (errors < 5)
                        printf("mismatch at [%d][%d] channel %d: got %lld, expected %lld\n",
                               out_row, out_col, out_channel,
                               (long long)output_image[out_row][out_col].channel[out_channel],
                               (long long)expected_image[out_row][out_col].channel[out_channel]);
                    errors++;
                }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n",
               sconv::out_height * sconv::out_width * sconv::out_channels);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
