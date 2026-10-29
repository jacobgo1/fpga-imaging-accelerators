#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// Stream to DDR: each pixel's CLASSES scores (SHIFT bits rounded off), then its
// class, the first highest score. One burst per patch row, the pixels inside the
// image only. Each burst's acknowledgement is collected BURSTS_IN_FLIGHT rows
// later instead of waited for (about 200 cycles a row on the ZCU104); the last
// ones at the end, so the kernel is done only when every word is in memory.
template<int SHIFT, int CLASSES, typename WORD_T>
WORD_T scores_and_class(const vec_t<acc_t, CLASSES>& sums) {
    #pragma HLS INLINE
    WORD_T word;
    int best = 0;
    for (int c = 0; c < CLASSES; c++) {
        #pragma HLS UNROLL
        word[c] = round_shift<SHIFT>(sums.v[c]);
    }
    for (int c = 1; c < CLASSES; c++) {
        #pragma HLS UNROLL
        if (word[c] > word[best]) best = c;
    }
    word[CLASSES] = best;
    return word;
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int SHIFT, int CLASSES, int BURSTS_IN_FLIGHT, typename WORD_T>
void store_rows(hls::stream<vec_t<acc_t, CLASSES> >& in, hls::burst_maxi<WORD_T> dout, int height, int width) {
    #pragma HLS INLINE
    const int rows = patch_rows<PATCH_HEIGHT>(height), cols = patch_cols<PATCH_WIDTH>(width);
    int unacknowledged = 0;
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int r = 0; r < PATCH_HEIGHT; r++) {
                const int y = pr * PATCH_HEIGHT + r, left = pc * PATCH_WIDTH;
                const int inside = y >= height ? 0 : width - left < PATCH_WIDTH ? width - left : PATCH_WIDTH;
                if (inside > 0) {
                    if (unacknowledged == BURSTS_IN_FLIGHT) dout.write_response();
                    else unacknowledged++;
                    dout.write_request(mul<false, uint32_t>((uint16_t)y, (uint16_t)width) + left, inside);
                }
                put: for (int x = 0; x < inside; x++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                    dout.write(scores_and_class<SHIFT, CLASSES, WORD_T>(in.read()));
                }
                drop: for (int x = inside; x < PATCH_WIDTH; x++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=31
                    in.read();
                }
            }
        }
    }
    drain: for (int k = 0; k < unacknowledged; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        dout.write_response();
    }
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int SHIFT, int CLASSES, typename WORD_T, int BURSTS_IN_FLIGHT = 8>
void store_scores(hls::stream<vec_t<acc_t, CLASSES> >& in, hls::burst_maxi<WORD_T> dout, int height, int width) {
    store_rows<PATCH_HEIGHT, PATCH_WIDTH, SHIFT, CLASSES, BURSTS_IN_FLIGHT>(in, dout, height, width);
}

// The same, then a word on finished once everything is in memory, for count_cycles.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int SHIFT, int CLASSES, typename WORD_T, int BURSTS_IN_FLIGHT = 8>
void store_scores(hls::stream<vec_t<acc_t, CLASSES> >& in, hls::burst_maxi<WORD_T> dout, int height, int width,
                  hls::stream<bool>& finished) {
    store_rows<PATCH_HEIGHT, PATCH_WIDTH, SHIFT, CLASSES, BURSTS_IN_FLIGHT>(in, dout, height, width);
    finished.write(true);
}
