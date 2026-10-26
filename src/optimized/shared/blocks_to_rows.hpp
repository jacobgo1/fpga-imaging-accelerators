#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// blocks_to_rows: a map in 2x2-block order (wino_conv3x3.hpp) back to raster
// order, one pixel a vector: each block row is two pixel rows. The top one
// goes out as the blocks come in, two pixels each; the bottom one is kept
// and goes out after it. One pixel a cycle.
//
// The image is cut into PH x PW patches (patch_grid.hpp); H x W is the map's
// size in a patch (pixels), H/2 x W/2 its blocks. T: the values' type.
template<int PH, int PW, int H, int W, int CH, typename T>
void blocks_to_rows(hls::stream<vec_t<T, 4 * CH> >& in, hls::stream<vec_t<T, CH> >& out,
                    int height, int width) {
    constexpr int GW = W / 2, ROW = 4 * GW;          // steps a block row: its 2 x W pixels
    constexpr int TRIPS = (H / 2) * ROW, MAX_TRIPS = 665 * TRIPS;
    (void)TRIPS; (void)MAX_TRIPS;   // only the pragma uses them
    T top_r[CH];                                     // the block's top right pixel
    #pragma HLS ARRAY_PARTITION variable=top_r type=complete
    T bottom_l[GW][CH], bottom_r[GW][CH];            // the block row's bottom pixels
    #pragma HLS ARRAY_PARTITION variable=bottom_l type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bottom_r type=complete dim=2
    const int n = patch_count<PH, PW>(height, width) * TRIPS;
    int s = 0;                                       // the step of the block row
    steps: for (int k = 0; k < n; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=TRIPS max=MAX_TRIPS
        #pragma HLS DEPENDENCE variable=bottom_l type=inter false
        #pragma HLS DEPENDENCE variable=bottom_r type=inter false
        vec_t<T, CH> o;
        if (s < 2 * GW && s % 2 == 0) {
            const vec_t<T, 4 * CH> v = in.read();
            for (int c = 0; c < CH; c++) {
                #pragma HLS UNROLL
                o.v[c] = v.v[c];
                top_r[c] = v.v[CH + c];
                bottom_l[s / 2][c] = v.v[2 * CH + c];
                bottom_r[s / 2][c] = v.v[3 * CH + c];
            }
        } else {
            const int x = s < 2 * GW ? 0 : (s - 2 * GW) / 2;
            for (int c = 0; c < CH; c++) {
                #pragma HLS UNROLL
                o.v[c] = s < 2 * GW ? top_r[c] : s % 2 == 0 ? bottom_l[x][c] : bottom_r[x][c];
            }
        }
        out.write(o);
        s = s < ROW - 1 ? s + 1 : 0;
    }
}
