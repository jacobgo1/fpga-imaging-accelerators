#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// torch.cat([first, second], dim=1) on two maps in raster order: each pixel's
// channels of first, then of second. One pixel a cycle. T: the values' type (both
// maps must have the same scale).
template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS, int FIRST_CHANNELS, int SECOND_CHANNELS, typename T = act_t>
void concat_pixels(hls::stream<vec_t<T, FIRST_CHANNELS> >& first,
                   hls::stream<vec_t<T, SECOND_CHANNELS> >& second,
                   hls::stream<vec_t<T, FIRST_CHANNELS + SECOND_CHANNELS> >& out, int height, int width) {
    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PIXELS;
    px: for (int k = 0; k < pixels; k++) {
        #pragma HLS PIPELINE II=1
        const vec_t<T, FIRST_CHANNELS> a = first.read();
        const vec_t<T, SECOND_CHANNELS> b = second.read();
        vec_t<T, FIRST_CHANNELS + SECOND_CHANNELS> joined;
        for (int ch = 0; ch < FIRST_CHANNELS; ch++) {
            #pragma HLS UNROLL
            joined.v[ch] = a.v[ch];
        }
        for (int ch = 0; ch < SECOND_CHANNELS; ch++) {
            #pragma HLS UNROLL
            joined.v[FIRST_CHANNELS + ch] = b.v[ch];
        }
        out.write(joined);
    }
}
