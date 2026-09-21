#include "streaming_relu.hpp"

void streaming_relu(
    const srelu_in_pixel_t  input_image [SRELU_HEIGHT][SRELU_WIDTH],
    srelu_out_pixel_t       output_image[SRELU_HEIGHT][SRELU_WIDTH])
{
    #pragma HLS INTERFACE mode=bram port=input_image
    #pragma HLS INTERFACE mode=bram port=output_image
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    constexpr data_t data_t_max = std::numeric_limits<data_t>::max();

    row_loop:
    for (int row = 0; row < SRELU_HEIGHT; row++) {
        col_loop:
        for (int col = 0; col < SRELU_WIDTH; col++) {
            #pragma HLS PIPELINE II=1

            srelu_out_pixel_t result;
            channel_loop:
            for (int channel = 0; channel < SRELU_CHANNELS; channel++) {
                #pragma HLS UNROLL
                const result_t value = input_image[row][col].channel[channel];

                // ReLU: negative becomes zero, everything else passes through.
                const result_t activated = (value > 0) ? value : result_t(0);

                // Narrow 64 bits down to 16, clamping instead of truncating --
                // see streaming_relu.hpp part 5 for why a plain cast would be
                // a silent correctness bug here.
                result.channel[channel] =
                    (activated > result_t(data_t_max)) ? data_t_max
                                                        : data_t(activated);
            }
            output_image[row][col] = result;
        }
    }
}
