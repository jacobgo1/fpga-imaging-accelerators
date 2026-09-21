#include "streaming_conv2d.hpp"

void streaming_conv2d(
    const sconv_in_pixel_t input_image [SCONV_IN_HEIGHT][SCONV_IN_WIDTH],
    const data_t           kernel_weights[SCONV_OUT_CHANNELS][SCONV_IN_CHANNELS]
                                         [SCONV_KERNEL_HEIGHT][SCONV_KERNEL_WIDTH],
    sconv_out_pixel_t      output_image[SCONV_OUT_HEIGHT][SCONV_OUT_WIDTH])
{
    #pragma HLS INTERFACE mode=bram port=input_image
    #pragma HLS INTERFACE mode=bram port=kernel_weights
    #pragma HLS INTERFACE mode=bram port=output_image
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // ---- weights: small, load the whole thing on-chip like conv2d does -
    data_t weight_buffer[SCONV_OUT_CHANNELS][SCONV_IN_CHANNELS]
                        [SCONV_KERNEL_HEIGHT][SCONV_KERNEL_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=0

    load_weights_out_channel:
    for (int out_channel = 0; out_channel < SCONV_OUT_CHANNELS; out_channel++)
        load_weights_in_channel:
        for (int in_channel = 0; in_channel < SCONV_IN_CHANNELS; in_channel++)
            load_weights_row:
            for (int kernel_row = 0; kernel_row < SCONV_KERNEL_HEIGHT; kernel_row++)
                load_weights_col:
                for (int kernel_col = 0; kernel_col < SCONV_KERNEL_WIDTH; kernel_col++) {
                    #pragma HLS PIPELINE II=1
                    weight_buffer[out_channel][in_channel][kernel_row][kernel_col] =
                        kernel_weights[out_channel][in_channel][kernel_row][kernel_col];
                }

    // ---- line buffer: the last SCONV_KERNEL_HEIGHT rows, and nothing else
    // Row banking is circular: physical bank = (logical row) % KERNEL_HEIGHT.
    // Once KERNEL_HEIGHT rows have streamed past, every bank holds exactly
    // one of the rows the current sliding window needs, and writing a new
    // row simply overwrites the row that has scrolled out of the window.
    sconv_in_pixel_t line_buffer[SCONV_KERNEL_HEIGHT][SCONV_IN_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buffer cyclic factor=SCONV_COL_BANKS dim=2
    // Reads and writes below hit different (row, col) addresses computed
    // from the same loop indices; the dependence checker cannot always
    // prove that by itself, so state it explicitly to keep II=1 reachable.
    #pragma HLS DEPENDENCE variable=line_buffer inter false

    stream_rows:
    for (int row = 0; row < SCONV_IN_HEIGHT; row++) {
        stream_cols:
        for (int col = 0; col < SCONV_IN_WIDTH; col++) {
            #pragma HLS PIPELINE II=1

            // One new pixel arrives, every cycle, unconditionally -- this
            // is the read that used to be "copy the whole image in" and is
            // now "look at each pixel exactly once, in the order it comes."
            const int row_bank = row % SCONV_KERNEL_HEIGHT;
            line_buffer[row_bank][col] = input_image[row][col];

            // The window needs KERNEL_HEIGHT rows and KERNEL_WIDTH columns
            // of history. Until both have accumulated, there is no full
            // window yet and nothing to compute -- just keep filling the
            // line buffer. (This is the fill-up period; see the walkthrough.)
            const bool have_enough_rows = row >= SCONV_KERNEL_HEIGHT - 1;
            const bool have_enough_cols = col >= SCONV_KERNEL_WIDTH  - 1;
            const bool row_on_stride = ((row - (SCONV_KERNEL_HEIGHT - 1)) % SCONV_STRIDE) == 0;
            const bool col_on_stride = ((col - (SCONV_KERNEL_WIDTH  - 1)) % SCONV_STRIDE) == 0;

            if (have_enough_rows && have_enough_cols && row_on_stride && col_on_stride) {
                const int out_row = (row - (SCONV_KERNEL_HEIGHT - 1)) / SCONV_STRIDE;
                const int out_col = (col - (SCONV_KERNEL_WIDTH  - 1)) / SCONV_STRIDE;

                sconv_out_pixel_t result;
                output_channel_loop:
                for (int out_channel = 0; out_channel < SCONV_OUT_CHANNELS; out_channel++) {
                    #pragma HLS UNROLL
                    result_t sum = 0;
                    for (int in_channel = 0; in_channel < SCONV_IN_CHANNELS; in_channel++) {
                        #pragma HLS UNROLL
                        for (int kernel_row = 0; kernel_row < SCONV_KERNEL_HEIGHT; kernel_row++) {
                            #pragma HLS UNROLL
                            for (int kernel_col = 0; kernel_col < SCONV_KERNEL_WIDTH; kernel_col++) {
                                #pragma HLS UNROLL
                                // Which line-buffer row this kernel row reads: the
                                // oldest row the window needs when kernel_row==0,
                                // sliding up to the row we just wrote this very
                                // cycle when kernel_row==KERNEL_HEIGHT-1.
                                const int window_row =
                                    (row - (SCONV_KERNEL_HEIGHT - 1) + kernel_row) % SCONV_KERNEL_HEIGHT;
                                const int window_col = col - (SCONV_KERNEL_WIDTH - 1) + kernel_col;
                                sum += static_cast<result_t>(
                                    line_buffer[window_row][window_col].channel[in_channel]
                                  * weight_buffer[out_channel][in_channel][kernel_row][kernel_col]);
                            }
                        }
                    }
                    result.channel[out_channel] = sum;
                }
                output_image[out_row][out_col] = result;
            }
        }
    }
}
