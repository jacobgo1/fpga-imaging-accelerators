#pragma once

#include <stdint.h>
#include "../common/hls_stream_shim.hpp"
#include "patch_grid.hpp"

// read_rows: DDR -> stream, the raw words of every PORTS-th row of each patch.
// din is the image, [height][width][WORDS] words (a word: some bands of a
// pixel, one bus beat). For each patch, rows PORT, PORT + PORTS, ... : PW
// pixels x WORDS words each, in memory order, one burst per row. PORTS of
// these, each on its own m_axi port, together read every row.
// Past the image's bottom the last row is read again; past its right edge the
// last pixel is repeated, so every patch is whole.
template<int PH, int PW, int WORDS, int PORTS, int PORT, typename WORD_T>
void read_rows(const WORD_T *din, hls::stream<WORD_T>& out, int height, int width) {
    static_assert(PH % PORTS == 0, "every port reads as many rows of a patch");
    constexpr int ROWS = PH / PORTS;
    const int rows = patch_rows<PH>(height), cols = patch_cols<PW>(width);
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int i = 0; i < ROWS; i++) {
                const int y = pr * PH + PORTS * i + PORT;
                const int x0 = pc * PW;
                const int n = width - x0 < PW ? width - x0 : PW;   // pixels inside the image
                const WORD_T *src = din + ((uint32_t)(y < height ? y : height - 1) * width + x0) * WORDS;
                WORD_T last[WORDS];                                // the last pixel read
                #pragma HLS ARRAY_PARTITION variable=last type=complete
                int w = 0;
                take: for (int k = 0; k < n * WORDS; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=15 max=480       // 15 words per pixel, 32 pixels
                    const WORD_T v = src[k];
                    out.write(v);
                    last[w] = v;
                    w = (w == WORDS - 1) ? 0 : w + 1;
                }
                edge: for (int k = 0; k < (PW - n) * WORDS; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=465
                    out.write(last[w]);
                    w = (w == WORDS - 1) ? 0 : w + 1;
                }
            }
        }
    }
}
