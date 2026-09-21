// streaming_relu_tb.cpp  --  testbench for streaming_relu, not a synthesis source
#include "streaming_relu.hpp"
#include <cstdio>
#include <limits>
#include <random>

static srelu_in_pixel_t  input_image   [srelu::height][srelu::width];
static srelu_out_pixel_t output_image  [srelu::height][srelu::width];
static srelu_out_pixel_t expected_image[srelu::height][srelu::width];

static void streaming_relu_reference()
{
    constexpr data_t data_t_max = std::numeric_limits<data_t>::max();
    for (int row = 0; row < srelu::height; row++)
        for (int col = 0; col < srelu::width; col++)
            for (int channel = 0; channel < srelu::channels; channel++) {
                const result_t value = input_image[row][col].channel[channel];
                const result_t activated = (value > 0) ? value : result_t(0);
                expected_image[row][col].channel[channel] =
                    (activated > result_t(data_t_max)) ? data_t_max : data_t(activated);
            }
}

int main()
{
    std::mt19937 rng(1);                              // fixed seed: repeatable
    // A wide distribution -- including values far past what data_t can hold
    // in either direction -- so the clamp path is actually exercised, not
    // just the "small value passes through unchanged" path.
    std::uniform_int_distribution<long long> dist(-4'000'000'000LL, 4'000'000'000LL);

    for (int row = 0; row < srelu::height; row++)
        for (int col = 0; col < srelu::width; col++)
            for (int channel = 0; channel < srelu::channels; channel++)
                input_image[row][col].channel[channel] = result_t(dist(rng));

    // A handful of hand-picked boundary values, overwriting the corner
    // pixel's channels: exactly the point where clamping starts to matter.
    if (srelu::channels >= 1) input_image[0][0].channel[0] = -1;                       // just negative -> 0
    if (srelu::channels >= 2) input_image[0][0].channel[1] = std::numeric_limits<data_t>::max();       // exact boundary -> unchanged
    if (srelu::channels >= 3) input_image[0][0].channel[2] = result_t(std::numeric_limits<data_t>::max()) + 1; // one past -> clamp

    streaming_relu_reference();
    streaming_relu(input_image, output_image);

    int errors = 0;
    for (int row = 0; row < srelu::height; row++)
        for (int col = 0; col < srelu::width; col++)
            for (int channel = 0; channel < srelu::channels; channel++)
                if (output_image[row][col].channel[channel] !=
                    expected_image[row][col].channel[channel]) {
                    if (errors < 5)
                        printf("mismatch at [%d][%d] channel %d: got %d, expected %d\n",
                               row, col, channel,
                               output_image[row][col].channel[channel],
                               expected_image[row][col].channel[channel]);
                    errors++;
                }

    if (errors == 0)
        printf("PASS  (%d outputs checked)\n", srelu::height * srelu::width * srelu::channels);
    else
        printf("FAIL  (%d mismatches)\n", errors);

    return errors == 0 ? 0 : 1;
}
