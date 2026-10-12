#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// store_scores: stream -> DDR, the last layer's sums of each patch into their
// place in the image. dout is [height][width][OUT] int32, the sums with SHIFT
// bits rounded off: OUT values per pixel and nothing else, so that the ARM
// reads the scores as one contiguous array (a fourth lane per pixel made its
// numpy float conversion 64 -> 260 ms). One int32 per cycle, OUT cycles per pixel; one
// burst per patch row, only the pixels inside the image; the rest of the row is
// read and dropped.
template<int PH, int PW, int SHIFT, int OUT>
void store_scores(hls::stream<vec_t<acc_t, OUT> >& in, int32_t *dout, int height, int width) {
    const int rows = patch_rows<PH>(height), cols = patch_cols<PW>(width);
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int r = 0; r < PH; r++) {
                const int y = pr * PH + r, x0 = pc * PW;
                const int n = y >= height ? 0 : width - x0 < PW ? width - x0 : PW;   // pixels to write
                int32_t *dst = dout + ((uint32_t)(y < height ? y : 0) * width + x0) * OUT;
                vec_t<acc_t, OUT> v;
                int c = 0;                      // the lane of v this cycle writes
                put: for (int k = 0; k < n * OUT; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=96
                    if (c == 0) v = in.read();
                    dst[k] = round_shift<SHIFT>(v.v[c]);
                    c = c == OUT - 1 ? 0 : c + 1;
                }
                drop: for (int x = n; x < PW; x++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=31
                    in.read();
                }
            }
        }
    }
}
