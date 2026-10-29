#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// concat_pixels for maps in 2x2-block order (wino_conv3x3.hpp): one block a cycle.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int FIRST_CHANNELS, int SECOND_CHANNELS>
void concat_blocks(hls::stream<vec_t<act_t, 4 * FIRST_CHANNELS> >& first,
                   hls::stream<vec_t<act_t, 4 * SECOND_CHANNELS> >& second,
                   hls::stream<vec_t<act_t, 4 * (FIRST_CHANNELS + SECOND_CHANNELS)> >& out, int height, int width) {
    constexpr int JOINED = FIRST_CHANNELS + SECOND_CHANNELS;
    constexpr int BLOCKS = (MAP_HEIGHT / 2) * (MAP_WIDTH / 2), MAX_BLOCKS = 665 * BLOCKS;
    (void)MAX_BLOCKS;   // only the pragma uses it
    const int blocks = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * BLOCKS;
    blocks: for (int k = 0; k < blocks; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=BLOCKS max=MAX_BLOCKS
        const vec_t<act_t, 4 * FIRST_CHANNELS> a = first.read();
        const vec_t<act_t, 4 * SECOND_CHANNELS> b = second.read();
        vec_t<act_t, 4 * JOINED> joined;
        for (int pixel = 0; pixel < 4; pixel++) {
            #pragma HLS UNROLL
            for (int ch = 0; ch < FIRST_CHANNELS; ch++) {
                #pragma HLS UNROLL
                joined.v[pixel * JOINED + ch] = a.v[pixel * FIRST_CHANNELS + ch];
            }
            for (int ch = 0; ch < SECOND_CHANNELS; ch++) {
                #pragma HLS UNROLL
                joined.v[pixel * JOINED + FIRST_CHANNELS + ch] = b.v[pixel * SECOND_CHANNELS + ch];
            }
        }
        out.write(joined);
    }
}
