#ifndef COMMON_LAYERS_POOL_HPP
#define COMMON_LAYERS_POOL_HPP

#include "types.hpp"

// ============================================================================
// Generic Streaming MaxPool2D Layer
//
// Parameters:
//   T           - Activation data type
//   IN_H, IN_W  - Spatial dimensions of the input feature map
//   CHANNELS    - Number of interleaved channels
//   POOL_SIZE   - Spatial pooling dimension (typically 2 for 2x2 pooling)
//   STRIDE      - Spatial stride (typically 2)
//
// Streaming Order:
//   Input:  HWC order (row -> col -> channel)
//   Output: HWC order (out_row -> out_col -> channel)
// ============================================================================
template <
    typename T,
    int IN_H,
    int IN_W,
    int CHANNELS,
    int POOL_SIZE = 2,
    int STRIDE = 2
>
void maxpool2d_layer(
    hls::stream<T>& in_stream,
    hls::stream<T>& out_stream
) {
    #pragma HLS INLINE off

    static_assert(STRIDE >= 1, "STRIDE must be >= 1");
    static_assert(IN_H >= POOL_SIZE && IN_W >= POOL_SIZE, "Pool size cannot exceed input");

    // Line buffer storing (POOL_SIZE - 1) previous rows
    T line_buf[POOL_SIZE - 1][IN_W][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=3

    // Window buffer
    T window[POOL_SIZE][POOL_SIZE][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    pool_row_loop:
    for (int r = 0; r < IN_H; ++r) {
        pool_col_loop:
        for (int c = 0; c < IN_W; ++c) {
            T current_pixel[CHANNELS];
            #pragma HLS ARRAY_PARTITION variable=current_pixel complete dim=1

            pool_in_ch:
            for (int ch = 0; ch < CHANNELS; ++ch) {
                #pragma HLS PIPELINE II=1
                current_pixel[ch] = in_stream.read();
            }

            // Shift window left
            pool_win_shift_r:
            for (int kr = 0; kr < POOL_SIZE; ++kr) {
                #pragma HLS UNROLL
                pool_win_shift_c:
                for (int kc = 0; kc < POOL_SIZE - 1; ++kc) {
                    #pragma HLS UNROLL
                    pool_win_shift_ch:
                    for (int ch = 0; ch < CHANNELS; ++ch) {
                        #pragma HLS UNROLL
                        window[kr][kc][ch] = window[kr][kc + 1][ch];
                    }
                }
            }

            // Insert new column
            pool_win_insert_kr:
            for (int kr = 0; kr < POOL_SIZE - 1; ++kr) {
                #pragma HLS UNROLL
                pool_win_insert_ch:
                for (int ch = 0; ch < CHANNELS; ++ch) {
                    #pragma HLS UNROLL
                    window[kr][POOL_SIZE - 1][ch] = line_buf[kr][c][ch];
                }
            }
            pool_win_insert_curr:
            for (int ch = 0; ch < CHANNELS; ++ch) {
                #pragma HLS UNROLL
                window[POOL_SIZE - 1][POOL_SIZE - 1][ch] = current_pixel[ch];
            }

            // Update line buffers
            pool_line_shift:
            for (int kr = 0; kr < POOL_SIZE - 2; ++kr) {
                #pragma HLS UNROLL
                pool_line_shift_ch:
                for (int ch = 0; ch < CHANNELS; ++ch) {
                    #pragma HLS UNROLL
                    line_buf[kr][c][ch] = line_buf[kr + 1][c][ch];
                }
            }
            pool_line_update:
            for (int ch = 0; ch < CHANNELS; ++ch) {
                #pragma HLS UNROLL
                if (POOL_SIZE > 1) {
                    line_buf[POOL_SIZE - 2][c][ch] = current_pixel[ch];
                }
            }

            // Check if window is ready at this stride sample point
            const bool row_valid = (r >= POOL_SIZE - 1) && ((r - (POOL_SIZE - 1)) % STRIDE == 0);
            const bool col_valid = (c >= POOL_SIZE - 1) && ((c - (POOL_SIZE - 1)) % STRIDE == 0);

            if (row_valid && col_valid) {
                pool_out_ch:
                for (int ch = 0; ch < CHANNELS; ++ch) {
                    #pragma HLS PIPELINE II=1
                    T max_val = window[0][0][ch];
                    pool_kr:
                    for (int kr = 0; kr < POOL_SIZE; ++kr) {
                        #pragma HLS UNROLL
                        pool_kc:
                        for (int kc = 0; kc < POOL_SIZE; ++kc) {
                            #pragma HLS UNROLL
                            if (window[kr][kc][ch] > max_val) {
                                max_val = window[kr][kc][ch];
                            }
                        }
                    }
                    out_stream.write(max_val);
                }
            }
        }
    }
}

#endif // COMMON_LAYERS_POOL_HPP
