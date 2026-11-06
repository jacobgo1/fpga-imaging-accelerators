#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// Every value of in to both a and b: a stream has one reader, so a map that two
// processes need (a U-Net encoder block's output, to the pool and to the skip)
// is copied here. T: the values' type.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int VALUES, int CHANNELS, typename T = act_t>
void duplicate(hls::stream<vec_t<T, CHANNELS> >& in, hls::stream<vec_t<T, CHANNELS> >& a,
               hls::stream<vec_t<T, CHANNELS> >& b, int height, int width) {
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        values: for (int i = 0; i < VALUES; i++) {
            #pragma HLS PIPELINE II=1
            const vec_t<T, CHANNELS> value = in.read();
            a.write(value);
            b.write(value);
        }
    }
}
