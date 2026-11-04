#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// torch.cat([first, second], dim=1) on two maps in raster order: each pixel's
// channels of first, then of second. One pixel a cycle.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS, int FIRST_CHANNELS, int SECOND_CHANNELS>
void concat_pixels(hls::stream<vec_t<act_t, FIRST_CHANNELS> >& first,
                   hls::stream<vec_t<act_t, SECOND_CHANNELS> >& second,
                   hls::stream<vec_t<act_t, FIRST_CHANNELS + SECOND_CHANNELS> >& out, int height, int width) {
    constexpr int MAX_PIXELS = 665 * PIXELS;
    (void)MAX_PIXELS;   // only the pragma uses it
    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PIXELS;
    px: for (int k = 0; k < pixels; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=PIXELS max=MAX_PIXELS
        const vec_t<act_t, FIRST_CHANNELS> a = first.read();
        const vec_t<act_t, SECOND_CHANNELS> b = second.read();
        vec_t<act_t, FIRST_CHANNELS + SECOND_CHANNELS> joined;
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
