#include "streaming_upsample.hpp"

void streaming_upsample(
    const supsample_pixel_t input_image [SUPSAMPLE_IN_HEIGHT][SUPSAMPLE_IN_WIDTH],
    supsample_pixel_t       output_image[SUPSAMPLE_OUT_HEIGHT][SUPSAMPLE_OUT_WIDTH])
{
    #pragma HLS INTERFACE mode=bram port=input_image
    #pragma HLS INTERFACE mode=bram port=output_image
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // One row, not several: everything needed to replay a row FACTOR
    // times without re-reading the input. No ARRAY_PARTITION here --
    // unlike conv2d/maxpool's line buffers, this is only ever touched one
    // element at a time (one read or one write per cycle), never several
    // elements at once, so there is no port-conflict to partition away.
    supsample_pixel_t row_buffer[SUPSAMPLE_IN_WIDTH];

    in_row_loop:
    for (int in_row = 0; in_row < SUPSAMPLE_IN_HEIGHT; in_row++) {
        // Every input row becomes FACTOR identical output rows.
        replicate_row_loop:
        for (int replicate_row = 0; replicate_row < SUPSAMPLE_FACTOR; replicate_row++) {
            // Holds the current input column's pixel across its FACTOR
            // repeats -- a small loop-carried register, same idea as
            // conv2d.cpp's row_partial_sums, just one element instead of
            // a whole row.
            supsample_pixel_t latched_pixel;

            out_col_loop:
            for (int out_col = 0; out_col < SUPSAMPLE_OUT_WIDTH; out_col++) {
                #pragma HLS PIPELINE II=1
                const int in_col        = out_col / SUPSAMPLE_FACTOR;
                const int replicate_col = out_col % SUPSAMPLE_FACTOR;

                // Only fetch/advance to a new pixel when starting a fresh
                // group of FACTOR output columns -- the other
                // FACTOR-1 output columns in the group just reuse it.
                if (replicate_col == 0) {
                    if (replicate_row == 0) {
                        // First pass over this input row: read for real,
                        // and remember it for the later replays below.
                        latched_pixel = input_image[in_row][in_col];
                        row_buffer[in_col] = latched_pixel;
                    } else {
                        // A later pass over the same input row: replay
                        // what was already read, no new input needed.
                        latched_pixel = row_buffer[in_col];
                    }
                }

                output_image[in_row * SUPSAMPLE_FACTOR + replicate_row][out_col] = latched_pixel;
            }
        }
    }
}
