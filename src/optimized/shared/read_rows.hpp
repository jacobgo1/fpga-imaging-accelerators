#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"
#include "patch_grid.hpp"

// DDR to stream: rows PORT, PORT + READ_PORTS, ... of each patch, WORDS_PER_PIXEL
// words a pixel, in memory order. Below the image the last row is read again,
// right of it the last pixel is repeated, so every patch is whole.
//
// Each row is requested in bursts of PIXELS_PER_BURST pixels, BURSTS_AHEAD bursts
// ahead of the reading, so the next row's address is on the bus while this one's
// words come in. Keep BURSTS_AHEAD + 1 within the port's num_read_outstanding.

// Where the burst at a request cursor starts in din, and how many words it has.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int READ_PORTS, int PORT, int PIXELS_PER_BURST>
void burst_at(int patch_row, int patch_col, int line, int burst, int height, int width,
              uint32_t& offset, int& length) {
    #pragma HLS INLINE
    const int y = patch_row * PATCH_HEIGHT + READ_PORTS * line + PORT;
    const int left = patch_col * PATCH_WIDTH;
    const int inside = width - left < PATCH_WIDTH ? width - left : PATCH_WIDTH;
    const int pixels = inside - burst * PIXELS_PER_BURST < PIXELS_PER_BURST ? inside - burst * PIXELS_PER_BURST
                                                                            : PIXELS_PER_BURST;
    const uint32_t row_start = mul<false, uint32_t>((uint16_t)(y < height ? y : height - 1), (uint16_t)width);
    offset = (row_start + left + burst * PIXELS_PER_BURST) * WORDS_PER_PIXEL;
    length = pixels * WORDS_PER_PIXEL;
}

// Moves the request cursor to the next burst; false past the last one.
template<int PATCH_WIDTH, int ROWS_PER_PORT, int PIXELS_PER_BURST>
bool next_burst(int& patch_row, int& patch_col, int& line, int& burst, int patch_rows, int patch_cols, int width) {
    #pragma HLS INLINE
    const int inside = width - patch_col * PATCH_WIDTH < PATCH_WIDTH ? width - patch_col * PATCH_WIDTH : PATCH_WIDTH;
    if (++burst < (inside + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST) return true;
    burst = 0;
    if (++line < ROWS_PER_PORT) return true;
    line = 0;
    if (++patch_col < patch_cols) return true;
    patch_col = 0;
    return ++patch_row < patch_rows;
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int READ_PORTS, int PORT,
         int PIXELS_PER_BURST = 16, int BURSTS_AHEAD = 2, typename WORD_T>
void read_rows_ahead(hls::burst_maxi<WORD_T> din, hls::stream<WORD_T>& out, int height, int width) {
    constexpr int ROWS_PER_PORT = PATCH_HEIGHT / READ_PORTS;
    const int rows = patch_rows<PATCH_HEIGHT>(height), cols = patch_cols<PATCH_WIDTH>(width);

    int request_row = 0, request_col = 0, request_line = 0, request_burst = 0;
    bool more = rows > 0 && cols > 0;
    uint32_t offset;
    int length;
    prime: for (int k = 0; k < BURSTS_AHEAD; k++) {
        if (more) {
            burst_at<PATCH_HEIGHT, PATCH_WIDTH, WORDS_PER_PIXEL, READ_PORTS, PORT, PIXELS_PER_BURST>(
                request_row, request_col, request_line, request_burst, height, width, offset, length);
            din.read_request(offset, length);
            more = next_burst<PATCH_WIDTH, ROWS_PER_PORT, PIXELS_PER_BURST>(
                request_row, request_col, request_line, request_burst, rows, cols, width);
        }
    }
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int i = 0; i < ROWS_PER_PORT; i++) {
                const int left = pc * PATCH_WIDTH;
                const int inside = width - left < PATCH_WIDTH ? width - left : PATCH_WIDTH;
                WORD_T last_pixel[WORDS_PER_PIXEL];
                #pragma HLS ARRAY_PARTITION variable=last_pixel type=complete
                int word = 0;
                burst: for (int s = 0; s * PIXELS_PER_BURST < inside; s++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=2
                    if (more) {
                        burst_at<PATCH_HEIGHT, PATCH_WIDTH, WORDS_PER_PIXEL, READ_PORTS, PORT, PIXELS_PER_BURST>(
                            request_row, request_col, request_line, request_burst, height, width, offset, length);
                        din.read_request(offset, length);
                        more = next_burst<PATCH_WIDTH, ROWS_PER_PORT, PIXELS_PER_BURST>(
                            request_row, request_col, request_line, request_burst, rows, cols, width);
                    }
                    const int pixels = inside - s * PIXELS_PER_BURST < PIXELS_PER_BURST ? inside - s * PIXELS_PER_BURST
                                                                                        : PIXELS_PER_BURST;
                    take: for (int k = 0; k < pixels * WORDS_PER_PIXEL; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=15 max=240
                        const WORD_T value = din.read();
                        out.write(value);
                        last_pixel[word] = value;
                        word = (word == WORDS_PER_PIXEL - 1) ? 0 : word + 1;
                    }
                }
                edge: for (int k = 0; k < (PATCH_WIDTH - inside) * WORDS_PER_PIXEL; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=465
                    out.write(last_pixel[word]);
                    word = (word == WORDS_PER_PIXEL - 1) ? 0 : word + 1;
                }
            }
        }
    }
}
