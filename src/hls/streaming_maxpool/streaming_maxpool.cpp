#include "streaming_maxpool.hpp"

void streaming_maxpool(
    const spool_pixel_t input_image [SPOOL_IN_HEIGHT][SPOOL_IN_WIDTH],
    spool_pixel_t       output_image[SPOOL_OUT_HEIGHT][SPOOL_OUT_WIDTH])
{
    #pragma HLS INTERFACE mode=bram port=input_image
    #pragma HLS INTERFACE mode=bram port=output_image
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // ---- line buffer: the last SPOOL_KERNEL_HEIGHT rows, and nothing else
    // Identical shelving trick to streaming_conv2d.cpp: physical bank =
    // (logical row) % KERNEL_HEIGHT, so KERNEL_HEIGHT rows recycle forever
    // regardless of how tall the image actually is. No weights this time --
    // there is nothing to load, pooling has no learned parameters at all.
    spool_pixel_t line_buffer[SPOOL_KERNEL_HEIGHT][SPOOL_IN_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buffer cyclic factor=SPOOL_COL_BANKS dim=2
    #pragma HLS DEPENDENCE variable=line_buffer inter false

    stream_rows:
    for (int row = 0; row < SPOOL_IN_HEIGHT; row++) {
        stream_cols:
        for (int col = 0; col < SPOOL_IN_WIDTH; col++) {
            #pragma HLS PIPELINE II=1

            const int row_bank = row % SPOOL_KERNEL_HEIGHT;
            line_buffer[row_bank][col] = input_image[row][col];

            const bool have_enough_rows = row >= SPOOL_KERNEL_HEIGHT - 1;
            const bool have_enough_cols = col >= SPOOL_KERNEL_WIDTH  - 1;
            const bool row_on_stride = ((row - (SPOOL_KERNEL_HEIGHT - 1)) % SPOOL_STRIDE) == 0;
            const bool col_on_stride = ((col - (SPOOL_KERNEL_WIDTH  - 1)) % SPOOL_STRIDE) == 0;

            if (have_enough_rows && have_enough_cols && row_on_stride && col_on_stride) {
                const int out_row = (row - (SPOOL_KERNEL_HEIGHT - 1)) / SPOOL_STRIDE;
                const int out_col = (col - (SPOOL_KERNEL_WIDTH  - 1)) / SPOOL_STRIDE;

                spool_pixel_t result;
                channel_loop:
                for (int channel = 0; channel < SPOOL_CHANNELS; channel++) {
                    #pragma HLS UNROLL
                    // Smallest possible value of data_t: anything real the
                    // window contains will beat it on the first comparison.
                    data_t largest = std::numeric_limits<data_t>::min();
                    for (int kernel_row = 0; kernel_row < SPOOL_KERNEL_HEIGHT; kernel_row++) {
                        #pragma HLS UNROLL
                        for (int kernel_col = 0; kernel_col < SPOOL_KERNEL_WIDTH; kernel_col++) {
                            #pragma HLS UNROLL
                            const int window_row =
                                (row - (SPOOL_KERNEL_HEIGHT - 1) + kernel_row) % SPOOL_KERNEL_HEIGHT;
                            const int window_col = col - (SPOOL_KERNEL_WIDTH - 1) + kernel_col;
                            const data_t candidate = line_buffer[window_row][window_col].channel[channel];
                            if (candidate > largest)
                                largest = candidate;
                        }
                    }
                    result.channel[channel] = largest;
                }
                output_image[out_row][out_col] = result;
            }
        }
    }
}
