#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// Every value of in to both a and b: a stream has one reader, so a map that two
// processes need (a U-Net encoder block's output, to the pool and to the skip)
// is copied here.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int VALUES, int CHANNELS>
void duplicate(hls::stream<vec_t<act_t, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& a,
               hls::stream<vec_t<act_t, CHANNELS> >& b, int height, int width) {
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        values: for (int i = 0; i < VALUES; i++) {
            #pragma HLS PIPELINE II=1
            const vec_t<act_t, CHANNELS> value = in.read();
            a.write(value);
            b.write(value);
        }
    }
}
