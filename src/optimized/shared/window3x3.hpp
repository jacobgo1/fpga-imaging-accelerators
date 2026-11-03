#pragma once

#include "fixed_point.hpp"

// The window of a streamed, same-padded 3x3 conv, in registers, and the two
// lines of input above it on chip. A new pixel at column col: that column of
// the lines moves up a row, the window shifts a column left, and rows r - 2,
// r - 1, r at col enter its right column. have: a pixel came in; past the last
// one, zeros enter.
template<int MAP_WIDTH, int CHANNELS, typename A_T>
void shift_window(A_T lines[2][MAP_WIDTH][CHANNELS], A_T window[3][3][CHANNELS], int& col,
                  const vec_t<A_T, CHANNELS>& pixel, bool have) {
    #pragma HLS INLINE
    for (int ch = 0; ch < CHANNELS; ch++) {
        #pragma HLS UNROLL
        const A_T two_up = lines[0][col][ch], one_up = lines[1][col][ch];
        const A_T now = have ? pixel.v[ch] : (A_T)0;
        lines[0][col][ch] = one_up;
        lines[1][col][ch] = now;
        for (int ky = 0; ky < 3; ky++) {
            #pragma HLS UNROLL
            window[ky][0][ch] = window[ky][1][ch];
            window[ky][1][ch] = window[ky][2][ch];
        }
        window[0][2][ch] = two_up;
        window[1][2][ch] = one_up;
        window[2][2][ch] = now;
    }
    col = (col == MAP_WIDTH - 1) ? 0 : col + 1;
}
