#pragma once

#include <stdint.h>
#include "../common/hls_stream_shim.hpp"
#include "patch_grid.hpp"

// read_patch_rows without the read-ahead: the same words in the same order, from
// a plain m_axi pointer. Each patch row is consecutive in memory, so HLS makes it
// one burst by itself; the cost is that every row waits for DDR's first word.
//
// DDR to stream: rows PORT, PORT + READ_PORTS, ... of each patch, WORDS_PER_PIXEL
// words a pixel, in memory order. Below the image the last row is read again,
// right of it the last pixel is repeated, so every patch is whole.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int READ_PORTS, int PORT, typename WORD_T>
void read_patch_rows_simple(const WORD_T* din, hls::stream<WORD_T>& out, int height, int width) {
    constexpr int ROWS_PER_PORT = PATCH_HEIGHT / READ_PORTS;
    const int rows = patch_rows<PATCH_HEIGHT>(height), cols = patch_cols<PATCH_WIDTH>(width);

    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            // This port's rows of the patch: line i is patch row READ_PORTS * i + PORT.
            line: for (int i = 0; i < ROWS_PER_PORT; i++) {
                const int y = pr * PATCH_HEIGHT + READ_PORTS * i + PORT;
                const int left = pc * PATCH_WIDTH;
                const int inside = width - left < PATCH_WIDTH ? width - left : PATCH_WIDTH;
                const uint32_t row_start = mul<false, uint32_t>((uint16_t)(y < height ? y : height - 1), (uint16_t)width);
                const WORD_T* row = din + (row_start + left) * WORDS_PER_PIXEL;
                WORD_T last_pixel[WORDS_PER_PIXEL];
                #pragma HLS ARRAY_PARTITION variable=last_pixel type=complete
                int word = 0;
                take: for (int k = 0; k < inside * WORDS_PER_PIXEL; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=15 max=480
                    const WORD_T value = row[k];
                    out.write(value);
                    last_pixel[word] = value;
                    word = (word == WORDS_PER_PIXEL - 1) ? 0 : word + 1;
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
