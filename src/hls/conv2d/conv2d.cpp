#include "conv2d.hpp"

void conv2d(const data_t input_image  [CONV_IN_CHANNELS][CONV_IN_HEIGHT][CONV_IN_WIDTH],
            const data_t kernel_weights[CONV_OUT_CHANNELS][CONV_IN_CHANNELS]
                                       [CONV_KERNEL_HEIGHT][CONV_KERNEL_WIDTH],
            result_t     output_image[CONV_OUT_CHANNELS][CONV_OUT_HEIGHT][CONV_OUT_WIDTH])
{
    #pragma HLS INTERFACE mode=bram port=input_image
    #pragma HLS INTERFACE mode=bram port=kernel_weights
    #pragma HLS INTERFACE mode=bram port=output_image
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // ---- local buffers -------------------------------------------------
    // On-chip copies of the interface arrays, partitioned so every value
    // a single cycle's math needs is in its own memory bank/register --
    // otherwise a BRAM's 1-2 read ports would force everything to wait in
    // line and II=1 would be impossible.
    data_t weight_buffer[CONV_OUT_CHANNELS][CONV_IN_CHANNELS]
                        [CONV_KERNEL_HEIGHT][CONV_KERNEL_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=0

    data_t input_buffer[CONV_IN_CHANNELS][CONV_IN_HEIGHT][CONV_IN_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=CONV_CHANNELS_PER_PASS dim=1
    #pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=CONV_IN_ROW_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=CONV_IN_COL_BANKS dim=3

    // ---- copy interface BRAMs into the partitioned local buffers -------
    // Plain, one-element-per-cycle copies. Nothing clever here -- this is
    // just getting the data from the caller's memory into the layout the
    // math below needs.
    load_weights_out_channel:
    for (int out_channel = 0; out_channel < CONV_OUT_CHANNELS; out_channel++)
        load_weights_in_channel:
        for (int in_channel = 0; in_channel < CONV_IN_CHANNELS; in_channel++)
            load_weights_row:
            for (int kernel_row = 0; kernel_row < CONV_KERNEL_HEIGHT; kernel_row++)
                load_weights_col:
                for (int kernel_col = 0; kernel_col < CONV_KERNEL_WIDTH; kernel_col++) {
                    #pragma HLS PIPELINE II=1
                    weight_buffer[out_channel][in_channel][kernel_row][kernel_col] =
                        kernel_weights[out_channel][in_channel][kernel_row][kernel_col];
                }

    load_input_channel:
    for (int in_channel = 0; in_channel < CONV_IN_CHANNELS; in_channel++)
        load_input_row:
        for (int in_row = 0; in_row < CONV_IN_HEIGHT; in_row++)
            load_input_col:
            for (int in_col = 0; in_col < CONV_IN_WIDTH; in_col++) {
                #pragma HLS PIPELINE II=1
                input_buffer[in_channel][in_row][in_col] = input_image[in_channel][in_row][in_col];
            }

    // ---- convolution ---------------------------------------------------
    // One entry per output column, holding the running total for that
    // column across passes over CONV_CHANNELS_PER_PASS-sized chunks of
    // input channels (see part 6 of conv2d.hpp).
    result_t row_partial_sums[CONV_OUT_WIDTH];
    #pragma HLS DEPENDENCE variable=row_partial_sums type=inter false

    output_channel_loop:
    for (int out_channel = 0; out_channel < CONV_OUT_CHANNELS; out_channel++) {
        output_row_loop:
        for (int out_row = 0; out_row < CONV_OUT_HEIGHT; out_row++) {
            // One pass = summing CONV_CHANNELS_PER_PASS input channels, for every
            // output column in this row, into row_partial_sums. Runs
            // CONV_IN_CHANNELS / CONV_CHANNELS_PER_PASS times so every input
            // channel gets included exactly once.
            channel_pass_loop:
            for (int channel_pass_start = 0; channel_pass_start < CONV_IN_CHANNELS;
                 channel_pass_start += CONV_CHANNELS_PER_PASS) {
                output_col_loop:
                for (int out_col = 0; out_col < CONV_OUT_WIDTH; out_col++) {
                    #pragma HLS PIPELINE II=1
                    // Top-left corner of the KERNEL_HEIGHT x KERNEL_WIDTH window
                    // in input_buffer that this output pixel is computed from.
                    const int in_row_start = out_row * CONV_STRIDE;
                    const int in_col_start = out_col * CONV_STRIDE;

                    // First pass starts a fresh sum; later passes pick up the
                    // running total this output column already has.
                    result_t pixel_sum = (channel_pass_start == 0) ? result_t(0)
                                                                    : row_partial_sums[out_col];

                    // Every (channel, kernel row, kernel col) triple in this pass
                    // is a separate multiply-add, all done in the same cycle.
                    for (int channel_in_pass = 0; channel_in_pass < CONV_CHANNELS_PER_PASS;
                         channel_in_pass++) {
                        #pragma HLS UNROLL
                        for (int kernel_row = 0; kernel_row < CONV_KERNEL_HEIGHT; kernel_row++) {
                            #pragma HLS UNROLL
                            for (int kernel_col = 0; kernel_col < CONV_KERNEL_WIDTH; kernel_col++) {
                                #pragma HLS UNROLL
                                // Multiply at 16x16 and widen afterwards. Casting both
                                // operands to result_t first asks for a 64x64 multiplier
                                // (~16 DSPs each, 27 of them here); the int promotion of
                                // two int16_t operands is exact -- |product| <= 2^30.
                                const int in_channel = channel_pass_start + channel_in_pass;
                                pixel_sum += static_cast<result_t>(
                                    input_buffer[in_channel][in_row_start + kernel_row]
                                                [in_col_start + kernel_col]
                                  * weight_buffer[out_channel][in_channel][kernel_row][kernel_col]);
                            }
                        }
                    }

                    row_partial_sums[out_col] = pixel_sum;
                    if (channel_pass_start + CONV_CHANNELS_PER_PASS >= CONV_IN_CHANNELS)
                        output_image[out_channel][out_row][out_col] = pixel_sum;
                }
            }
        }
    }
}
