// conv2d_tb.cpp  --  testbench for the conv2d kernel, not a synthesis source
#include "conv2d.hpp"
#include <cstdio>
#include <random>

static data_t   input_image   [conv::in_channels][conv::in_height][conv::in_width];
static data_t   kernel_weights[conv::out_channels][conv::in_channels]
                              [conv::kernel_height][conv::kernel_width];
static result_t output_image  [conv::out_channels][conv::out_height][conv::out_width];
static result_t expected_image[conv::out_channels][conv::out_height][conv::out_width];

// Plain, obviously-correct reference. No pragmas, no cleverness.
static void conv2d_reference()
{
    for (int out_channel = 0; out_channel < conv::out_channels; out_channel++)
        for (int out_row = 0; out_row < conv::out_height; out_row++)
            for (int out_col = 0; out_col < conv::out_width; out_col++) {
                result_t sum = 0;
                for (int in_channel = 0; in_channel < conv::in_channels; in_channel++)
                    for (int kernel_row = 0; kernel_row < conv::kernel_height; kernel_row++)
                        for (int kernel_col = 0; kernel_col < conv::kernel_width; kernel_col++)
                            sum += result_t(input_image[in_channel]
                                                        [out_row * conv::stride + kernel_row]
                                                        [out_col * conv::stride + kernel_col])
                                 * result_t(kernel_weights[out_channel][in_channel]
                                                          [kernel_row][kernel_col]);
                expected_image[out_channel][out_row][out_col] = sum;
            }
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable
    std::uniform_int_distribution<int> dist(-100, 100);

    for (int in_channel = 0; in_channel < conv::in_channels; in_channel++)
        for (int in_row = 0; in_row < conv::in_height; in_row++)
            for (int in_col = 0; in_col < conv::in_width; in_col++)
                input_image[in_channel][in_row][in_col] = data_t(dist(rng));

    for (int out_channel = 0; out_channel < conv::out_channels; out_channel++)
        for (int in_channel = 0; in_channel < conv::in_channels; in_channel++)
            for (int kernel_row = 0; kernel_row < conv::kernel_height; kernel_row++)
                for (int kernel_col = 0; kernel_col < conv::kernel_width; kernel_col++)
                    kernel_weights[out_channel][in_channel][kernel_row][kernel_col] =
                        data_t(dist(rng));

    conv2d_reference();
    conv2d(input_image, kernel_weights, output_image);

    int errors = 0;
    for (int out_channel = 0; out_channel < conv::out_channels; out_channel++)
        for (int out_row = 0; out_row < conv::out_height; out_row++)
            for (int out_col = 0; out_col < conv::out_width; out_col++)
                if (output_image[out_channel][out_row][out_col] !=
                    expected_image[out_channel][out_row][out_col]) {
                    if (errors < 5)
                        printf("mismatch at [%d][%d][%d]: got %lld, expected %lld\n",
                               out_channel, out_row, out_col,
                               (long long)output_image[out_channel][out_row][out_col],
                               (long long)expected_image[out_channel][out_row][out_col]);
                    errors++;
                }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n",
               conv::out_channels * conv::out_height * conv::out_width);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
