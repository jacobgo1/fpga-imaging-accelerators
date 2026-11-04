#pragma once

#include <stdint.h>
#include "../common/hls_stream_shim.hpp"
#include "../common/hls_burst_maxi_shim.hpp"
#include "patch_grid.hpp"

// DDR to stream: rows PORT, PORT + READ_PORTS, ... of each patch, WORDS_PER_PIXEL
// words a pixel, in memory order. Below the image the last row is read again,
// right of it the last pixel is repeated, so every patch is whole.
//
// Each row is requested in bursts of PIXELS_PER_BURST pixels, BURSTS_AHEAD bursts
// ahead of the reading, so the next row's address is on the bus while this one's
// words come in. Keep BURSTS_AHEAD + 1 within the port's num_read_outstanding.
//
// Two walks over the same bursts: the request cursor (request_*) asks DDR for a
// burst, the read loops (pr, pc, i, s) take its words BURSTS_AHEAD bursts later.

// Where the burst at a request cursor starts in din, and how many words it has.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int READ_PORTS, int PORT, int PIXELS_PER_BURST>
void burst_at(int patch_row, int patch_col, int line, int burst, int height, int width,
              uint32_t& offset, int& length) {
    #pragma HLS INLINE
    const int y = patch_row * PATCH_HEIGHT + READ_PORTS * line + PORT;            // image row of this line
    const int left = patch_col * PATCH_WIDTH;                                      // image column of the patch
    const int inside = width - left < PATCH_WIDTH ? width - left : PATCH_WIDTH;    // pixels of the row in the image
    const int pixels = inside - burst * PIXELS_PER_BURST < PIXELS_PER_BURST ? inside - burst * PIXELS_PER_BURST
                                                                            : PIXELS_PER_BURST;  // this burst's pixels
    const uint32_t row_start = mul<false, uint32_t>((uint16_t)(y < height ? y : height - 1), (uint16_t)width);  // below: last row
    offset = (row_start + left + burst * PIXELS_PER_BURST) * WORDS_PER_PIXEL;     // first word of the burst
    length = pixels * WORDS_PER_PIXEL;                                             // words in the burst
}

// Moves the request cursor to the next burst; false past the last one.
// It counts like an odometer: burst, then line, then patch column, then patch row.
template<int PATCH_WIDTH, int ROWS_PER_PORT, int PIXELS_PER_BURST>
bool next_burst(int& patch_row, int& patch_col, int& line, int& burst, int patch_rows, int patch_cols, int width) {
    #pragma HLS INLINE
    const int inside = width - patch_col * PATCH_WIDTH < PATCH_WIDTH ? width - patch_col * PATCH_WIDTH : PATCH_WIDTH;
    if (++burst < (inside + PIXELS_PER_BURST - 1) / PIXELS_PER_BURST) return true;  // next burst of this row
    burst = 0;
    if (++line < ROWS_PER_PORT) return true;                                          // next row of this patch
    line = 0;
    if (++patch_col < patch_cols) return true;                                        // next patch to the right
    patch_col = 0;
    return ++patch_row < patch_rows;                                                  // next row of patches, or done
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int READ_PORTS, int PORT,
         int PIXELS_PER_BURST = 16, int BURSTS_AHEAD = 2, typename WORD_T>
void read_patch_rows(hls::burst_maxi<WORD_T> din, hls::stream<WORD_T>& out, int height, int width) {
    constexpr int ROWS_PER_PORT = PATCH_HEIGHT / READ_PORTS;   // rows of a patch this port reads
    const int rows = patch_rows<PATCH_HEIGHT>(height), cols = patch_cols<PATCH_WIDTH>(width);

    // The request cursor: the next burst to ask DDR for.
    int request_row = 0, request_col = 0, request_line = 0, request_burst = 0;
    bool more = rows > 0 && cols > 0;   // false once every burst is requested
    uint32_t offset;
    int length;

    // Read-ahead, start: ask for the first BURSTS_AHEAD bursts before reading any.
    prime: for (int k = 0; k < BURSTS_AHEAD; k++) {
        if (more) {
            burst_at<PATCH_HEIGHT, PATCH_WIDTH, WORDS_PER_PIXEL, READ_PORTS, PORT, PIXELS_PER_BURST>(
                request_row, request_col, request_line, request_burst, height, width, offset, length);
            din.read_request(offset, length);   // request: DDR starts fetching, the words come later
            more = next_burst<PATCH_WIDTH, ROWS_PER_PORT, PIXELS_PER_BURST>(
                request_row, request_col, request_line, request_burst, rows, cols, width);
        }
    }

    // The reading: patch by patch, this port's rows of each, a burst at a time.
    patch_row: for (int pr = 0; pr < rows; pr++) {                // each row of patches
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {            // each patch in it
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int i = 0; i < ROWS_PER_PORT; i++) {       // this port's rows: patch row READ_PORTS * i + PORT
                const int left = pc * PATCH_WIDTH;
                const int inside = width - left < PATCH_WIDTH ? width - left : PATCH_WIDTH;  // pixels in the image
                WORD_T last_pixel[WORDS_PER_PIXEL];               // the last pixel read, for the right edge
                #pragma HLS ARRAY_PARTITION variable=last_pixel type=complete
                int word = 0;                                     // which word of the pixel comes next
                burst: for (int s = 0; s * PIXELS_PER_BURST < inside; s++) {  // each burst of the row
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=2
                    // Read-ahead, steady state: one new request for every burst read,
                    // so BURSTS_AHEAD bursts stay requested ahead of the reading.
                    if (more) {
                        burst_at<PATCH_HEIGHT, PATCH_WIDTH, WORDS_PER_PIXEL, READ_PORTS, PORT, PIXELS_PER_BURST>(
                            request_row, request_col, request_line, request_burst, height, width, offset, length);
                        din.read_request(offset, length);         // request: a burst BURSTS_AHEAD ahead of this one
                        more = next_burst<PATCH_WIDTH, ROWS_PER_PORT, PIXELS_PER_BURST>(
                            request_row, request_col, request_line, request_burst, rows, cols, width);
                    }
                    const int pixels = inside - s * PIXELS_PER_BURST < PIXELS_PER_BURST ? inside - s * PIXELS_PER_BURST
                                                                                        : PIXELS_PER_BURST;
                    take: for (int k = 0; k < pixels * WORDS_PER_PIXEL; k++) {  // this burst's words, one a cycle
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=15 max=240
                        const WORD_T value = din.read();          // read: the next word of the oldest request
                        out.write(value);                         // write: to patch_buffer
                        last_pixel[word] = value;                 // keep it, in case the edge repeats it
                        word = (word == WORDS_PER_PIXEL - 1) ? 0 : word + 1;
                    }
                }
                edge: for (int k = 0; k < (PATCH_WIDTH - inside) * WORDS_PER_PIXEL; k++) {  // right of the image
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=465
                    out.write(last_pixel[word]);                  // write: the last pixel again, no DDR read
                    word = (word == WORDS_PER_PIXEL - 1) ? 0 : word + 1;
                }
            }
        }
    }
}
