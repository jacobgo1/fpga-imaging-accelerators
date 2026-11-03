#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// 2x2 max pool, raster order in and out. A pooled pixel leaves as soon as its
// block's last value arrives.
template<int MAP_HEIGHT, int MAP_WIDTH, int CHANNELS>
void max_pool_patch(hls::stream<vec_t<act_t, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out) {
    act_t even_row[MAP_WIDTH / 2][CHANNELS];   // the even row, pooled in pairs
    #pragma HLS ARRAY_PARTITION variable=even_row type=complete dim=2
    act_t left[CHANNELS];                      // the even column's value
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    for (int y = 0; y < MAP_HEIGHT; y++)
        for (int x = 0; x < MAP_WIDTH; x++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=even_row type=inter false
            const vec_t<act_t, CHANNELS> value = in.read();
            vec_t<act_t, CHANNELS> pooled;
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                const act_t a = value.v[ch];
                if (x % 2 == 0) {
                    left[ch] = a;
                } else {
                    const act_t pair = max2(left[ch], a);
                    if (y % 2 == 0) even_row[x / 2][ch] = pair;
                    else pooled.v[ch] = max2(even_row[x / 2][ch], pair);
                }
            }
            if (y % 2 == 1 && x % 2 == 1) out.write(pooled);
        }
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int CHANNELS>
void max_pool(hls::stream<vec_t<act_t, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out,
              int height, int width) {
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        max_pool_patch<MAP_HEIGHT, MAP_WIDTH>(in, out);
    }
}
