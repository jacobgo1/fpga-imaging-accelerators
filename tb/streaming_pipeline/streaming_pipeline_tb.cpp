// streaming_pipeline_tb.cpp  --  testbench for streaming_pipeline, not a
// synthesis source. Reference model is deliberately just the three
// existing reference models (conv2d, relu, maxpool textbook math) chained
// through plain intermediate arrays -- proving the DATAFLOW/hls::stream
// version computes the same thing as the three separate stages would, not
// re-deriving what "correct" means from scratch.
#include "streaming_pipeline.hpp"
#include <cstdio>
#include <limits>
#include <random>

static sconv_in_pixel_t  input_image   [spipe::in_height][spipe::in_width];
static data_t            kernel_weights[spipe::out_channels][spipe::in_channels]
                                       [spipe::kernel_height][spipe::kernel_width];
static spool_pixel_t     output_image  [spipe::out_height][spipe::out_width];

static sconv_out_pixel_t  expected_conv[spipe::conv_out_height][spipe::conv_out_width];
static srelu_out_pixel_t  expected_relu[spipe::conv_out_height][spipe::conv_out_width];
static spool_pixel_t      expected_pool[spipe::out_height][spipe::out_width];

static void reference_conv()
{
    for (int out_row = 0; out_row < spipe::conv_out_height; out_row++)
        for (int out_col = 0; out_col < spipe::conv_out_width; out_col++)
            for (int out_channel = 0; out_channel < spipe::out_channels; out_channel++) {
                result_t sum = 0;
                for (int in_channel = 0; in_channel < spipe::in_channels; in_channel++)
                    for (int kernel_row = 0; kernel_row < spipe::kernel_height; kernel_row++)
                        for (int kernel_col = 0; kernel_col < spipe::kernel_width; kernel_col++)
                            sum += result_t(input_image[out_row * spipe::stride + kernel_row]
                                                        [out_col * spipe::stride + kernel_col]
                                                        .channel[in_channel])
                                 * result_t(kernel_weights[out_channel][in_channel]
                                                          [kernel_row][kernel_col]);
                expected_conv[out_row][out_col].channel[out_channel] = sum;
            }
}

static void reference_relu()
{
    constexpr data_t data_t_max = std::numeric_limits<data_t>::max();
    for (int row = 0; row < spipe::conv_out_height; row++)
        for (int col = 0; col < spipe::conv_out_width; col++)
            for (int channel = 0; channel < spipe::out_channels; channel++) {
                const result_t value = expected_conv[row][col].channel[channel];
                const result_t activated = (value > 0) ? value : result_t(0);
                expected_relu[row][col].channel[channel] =
                    (activated > result_t(data_t_max)) ? data_t_max : data_t(activated);
            }
}

static void reference_pool()
{
    for (int out_row = 0; out_row < spipe::out_height; out_row++)
        for (int out_col = 0; out_col < spipe::out_width; out_col++)
            for (int channel = 0; channel < spipe::out_channels; channel++) {
                data_t largest = std::numeric_limits<data_t>::min();
                for (int kernel_row = 0; kernel_row < spipe::pool_kernel_height; kernel_row++)
                    for (int kernel_col = 0; kernel_col < spipe::pool_kernel_width; kernel_col++) {
                        const data_t candidate =
                            expected_relu[out_row * spipe::pool_stride + kernel_row]
                                         [out_col * spipe::pool_stride + kernel_col]
                                         .channel[channel];
                        if (candidate > largest)
                            largest = candidate;
                    }
                expected_pool[out_row][out_col].channel[channel] = largest;
            }
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable
    std::uniform_int_distribution<int> dist(-100, 100);

    for (int in_row = 0; in_row < spipe::in_height; in_row++)
        for (int in_col = 0; in_col < spipe::in_width; in_col++)
            for (int in_channel = 0; in_channel < spipe::in_channels; in_channel++)
                input_image[in_row][in_col].channel[in_channel] = data_t(dist(rng));

    for (int out_channel = 0; out_channel < spipe::out_channels; out_channel++)
        for (int in_channel = 0; in_channel < spipe::in_channels; in_channel++)
            for (int kernel_row = 0; kernel_row < spipe::kernel_height; kernel_row++)
                for (int kernel_col = 0; kernel_col < spipe::kernel_width; kernel_col++)
                    kernel_weights[out_channel][in_channel][kernel_row][kernel_col] =
                        data_t(dist(rng));

    reference_conv();
    reference_relu();
    reference_pool();

    streaming_pipeline(input_image, kernel_weights, output_image);

    int errors = 0;
    for (int out_row = 0; out_row < spipe::out_height; out_row++)
        for (int out_col = 0; out_col < spipe::out_width; out_col++)
            for (int out_channel = 0; out_channel < spipe::out_channels; out_channel++)
                if (output_image[out_row][out_col].channel[out_channel] !=
                    expected_pool[out_row][out_col].channel[out_channel]) {
                    if (errors < 5)
                        printf("mismatch at [%d][%d] channel %d: got %d, expected %d\n",
                               out_row, out_col, out_channel,
                               (int)output_image[out_row][out_col].channel[out_channel],
                               (int)expected_pool[out_row][out_col].channel[out_channel]);
                    errors++;
                }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n",
               spipe::out_height * spipe::out_width * spipe::out_channels);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
