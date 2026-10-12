#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// store_scores: stream -> DDR, the last layer's sums of each patch into their
// place in the image. dout is [height][width] words of OUT + 1 lanes: the OUT
// scores (the sums with SHIFT bits rounded off), then the class, the first
// highest score (like numpy's argmax). A word is one bus beat. One burst per
// patch row, only the pixels inside the image; the rest of the row is read
// and dropped.
//
// The memory answers each burst with an acknowledgement (AXI's write
// response) once its words are in DDR. Written as a plain pointer (dout[i] = v
// in a loop), HLS waits for that answer at the end of every burst, before the
// next row's burst may start -- on the ZCU104 about 200 cycles a row, for 32
// words of data. So the bursts are spelled out (hls::burst_maxi): request,
// words, and the acknowledgement collected IN_FLIGHT rows later, long after it
// arrived, while the next rows already go out. The last ones are collected at
// the end, so the kernel reports done only when every word is in memory.
template<int SHIFT, int OUT, typename OUT_T>
OUT_T scores_and_class(const vec_t<acc_t, OUT>& v) {
    #pragma HLS INLINE
    OUT_T o;
    int best = 0;
    for (int c = 0; c < OUT; c++) {
        #pragma HLS UNROLL
        o[c] = round_shift<SHIFT>(v.v[c]);
    }
    for (int c = 1; c < OUT; c++) {
        #pragma HLS UNROLL
        if (o[c] > o[best]) best = c;
    }
    o[OUT] = best;
    return o;
}

// IN_FLIGHT: bursts allowed to wait for their acknowledgement; at most the
// m_axi port's num_write_outstanding.
template<int PH, int PW, int SHIFT, int OUT, typename OUT_T, int IN_FLIGHT = 8>
void store_scores(hls::stream<vec_t<acc_t, OUT> >& in, hls::burst_maxi<OUT_T> dout, int height, int width) {
    const int rows = patch_rows<PH>(height), cols = patch_cols<PW>(width);
    int pending = 0;                            // bursts not yet acknowledged
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int r = 0; r < PH; r++) {
                const int y = pr * PH + r, x0 = pc * PW;
                const int n = y >= height ? 0 : width - x0 < PW ? width - x0 : PW;   // pixels to write
                if (n > 0) {
                    // Wait for the oldest burst only when IN_FLIGHT are out.
                    if (pending == IN_FLIGHT) dout.write_response();
                    else pending++;
                    dout.write_request((uint64_t)y * width + x0, n);
                }
                put: for (int x = 0; x < n; x++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                    dout.write(scores_and_class<SHIFT, OUT, OUT_T>(in.read()));
                }
                drop: for (int x = n; x < PW; x++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=31
                    in.read();
                }
            }
        }
    }
    drain: for (int k = 0; k < pending; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        dout.write_response();
    }
}
