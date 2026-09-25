#ifndef COMMON_LAYERS_CONV2D_HPP
#define COMMON_LAYERS_CONV2D_HPP

#include "types.hpp"
#include "line_buffer.hpp"

// ============================================================================
// Generic Streaming Conv2D Layer
//
// Parameters:
//   T_IN         - Input activation data type (e.g. data_t / int16_t)
//   T_WEIGHT     - Weight data type (e.g. data_t / int16_t)
//   T_ACC        - Accumulator data type (e.g. acc_t / int32_t)
//   T_OUT        - Output data type (e.g. result_t / int16_t)
//   IN_H, IN_W   - Spatial dimensions of the input feature map
//   IN_CH        - Number of input channels
//   OUT_CH       - Number of output feature maps / filters
//   K_SIZE       - Square kernel size (e.g. 3 for 3x3)
//   STRIDE       - Stride (default 1)
//   APPLY_RELU   - If true, fuses ReLU activation into the layer output
//
// Streaming Order:
//   Pixels are streamed in HWC order: row -> col -> in_channel.
//   Output pixels are written in HWC order: out_row -> out_col -> out_channel.
// ============================================================================
template <
    typename T_IN,
    typename T_WEIGHT,
    typename T_ACC,
    typename T_OUT,
    int IN_H,
    int IN_W,
    int IN_CH,
    int OUT_CH,
    int K_SIZE,
    int STRIDE = 1,
    bool APPLY_RELU = false
>
void conv2d_layer(
    hls::stream<T_IN>& in_stream,
    const T_WEIGHT weights[OUT_CH][IN_CH][K_SIZE][K_SIZE],
    const T_WEIGHT bias[OUT_CH],
    hls::stream<T_OUT>& out_stream
) {
    #pragma HLS INLINE off

    static_assert(STRIDE >= 1, "STRIDE must be >= 1");
    static_assert(IN_H >= K_SIZE && IN_W >= K_SIZE, "Kernel cannot be larger than input");

    // Line buffer: stores (K_SIZE - 1) previous rows for all input channels
    T_IN line_buf[K_SIZE - 1][IN_W][IN_CH];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=3

    // 2D sliding window: stores K_SIZE x K_SIZE patch for all input channels
    T_IN window[K_SIZE][K_SIZE][IN_CH];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // Stream through every input pixel
    row_loop:
    for (int r = 0; r < IN_H; ++r) {
        col_loop:
        for (int c = 0; c < IN_W; ++c) {
            // Read input channels for the current pixel (r, c)
            T_IN current_pixel[IN_CH];
            #pragma HLS ARRAY_PARTITION variable=current_pixel complete dim=1

            in_ch_read:
            for (int ch = 0; ch < IN_CH; ++ch) {
                #pragma HLS PIPELINE II=1
                current_pixel[ch] = in_stream.read();
            }

            // Shift 2D sliding window to the left
            win_shift_r:
            for (int kr = 0; kr < K_SIZE; ++kr) {
                #pragma HLS UNROLL
                win_shift_c:
                for (int kc = 0; kc < K_SIZE - 1; ++kc) {
                    #pragma HLS UNROLL
                    win_shift_ch:
                    for (int ch = 0; ch < IN_CH; ++ch) {
                        #pragma HLS UNROLL
                        window[kr][kc][ch] = window[kr][kc + 1][ch];
                    }
                }
            }

            // Insert new column into sliding window from line buffers and incoming pixel
            win_insert_kr:
            for (int kr = 0; kr < K_SIZE - 1; ++kr) {
                #pragma HLS UNROLL
                win_insert_ch:
                for (int ch = 0; ch < IN_CH; ++ch) {
                    #pragma HLS UNROLL
                    window[kr][K_SIZE - 1][ch] = line_buf[kr][c][ch];
                }
            }
            win_insert_current:
            for (int ch = 0; ch < IN_CH; ++ch) {
                #pragma HLS UNROLL
                window[K_SIZE - 1][K_SIZE - 1][ch] = current_pixel[ch];
            }

            // Update line buffers with current pixel
            line_buf_shift:
            for (int kr = 0; kr < K_SIZE - 2; ++kr) {
                #pragma HLS UNROLL
                line_buf_shift_ch:
                for (int ch = 0; ch < IN_CH; ++ch) {
                    #pragma HLS UNROLL
                    line_buf[kr][c][ch] = line_buf[kr + 1][c][ch];
                }
            }
            line_buf_update_ch:
            for (int ch = 0; ch < IN_CH; ++ch) {
                #pragma HLS UNROLL
                if (K_SIZE > 1) {
                    line_buf[K_SIZE - 2][c][ch] = current_pixel[ch];
                }
            }

            // Check if sliding window has filled valid image data
            const bool row_valid = (r >= K_SIZE - 1) && ((r - (K_SIZE - 1)) % STRIDE == 0);
            const bool col_valid = (c >= K_SIZE - 1) && ((c - (K_SIZE - 1)) % STRIDE == 0);

            if (row_valid && col_valid) {
                // Compute all output channels for this valid spatial window
                out_ch_loop:
                for (int out_c = 0; out_c < OUT_CH; ++out_c) {
                    #pragma HLS PIPELINE II=1
                    T_ACC acc = (bias != nullptr) ? static_cast<T_ACC>(bias[out_c]) : T_ACC(0);

                    // Unrolled dot product over kernel window and input channels
                    mac_in_ch:
                    for (int in_c = 0; in_c < IN_CH; ++in_c) {
                        #pragma HLS UNROLL
                        mac_kr:
                        for (int kr = 0; kr < K_SIZE; ++kr) {
                            #pragma HLS UNROLL
                            mac_kc:
                            for (int kc = 0; kc < K_SIZE; ++kc) {
                                #pragma HLS UNROLL
                                acc += static_cast<T_ACC>(window[kr][kc][in_c]) *
                                       static_cast<T_ACC>(weights[out_c][in_c][kr][kc]);
                            }
                        }
                    }

                    if (APPLY_RELU && acc < T_ACC(0)) {
                        acc = T_ACC(0);
                    }

                    out_stream.write(static_cast<T_OUT>(acc));
                }
            }
        }
    }
}

#endif // COMMON_LAYERS_CONV2D_HPP
