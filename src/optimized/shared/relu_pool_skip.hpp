#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// relu_pool that also sends every activation to skip: a U-Net encoder block's
// output goes both to the next block and to the skip connection. Activations
// a conv already ReLU'd pass as they are (SHIFT 0, see relu_in).
template<int MAP_HEIGHT, int MAP_WIDTH, int SHIFT, int CHANNELS, typename IN_T>
void relu_pool_skip_patch(hls::stream<vec_t<IN_T, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& skip,
                          hls::stream<vec_t<act_t, CHANNELS> >& out) {
    act_t even_row[MAP_WIDTH / 2][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=even_row type=complete dim=2
    act_t left[CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    for (int y = 0; y < MAP_HEIGHT; y++)
        for (int x = 0; x < MAP_WIDTH; x++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=even_row type=inter false
            const vec_t<IN_T, CHANNELS> value = in.read();
            vec_t<act_t, CHANNELS> activations, pooled;
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                activations.v[ch] = relu_in<SHIFT>(value.v[ch]);
                if (x % 2 == 0) {
                    left[ch] = activations.v[ch];
                } else {
                    const act_t pair = max2(left[ch], activations.v[ch]);
                    if (y % 2 == 0) even_row[x / 2][ch] = pair;
                    else pooled.v[ch] = max2(even_row[x / 2][ch], pair);
                }
            }
            skip.write(activations);
            if (y % 2 == 1 && x % 2 == 1) out.write(pooled);
        }
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int SHIFT, int CHANNELS, typename IN_T>
void relu_pool_skip(hls::stream<vec_t<IN_T, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& skip,
                    hls::stream<vec_t<act_t, CHANNELS> >& out, int height, int width) {
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_pool_skip_patch<MAP_HEIGHT, MAP_WIDTH, SHIFT>(in, skip, out);
    }
}
