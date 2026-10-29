#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// ReLU, round off SHIFT bits, 2x2 max pool, upsample by 2: every 2x2 block
// becomes its maximum, same size. Each pair of rows is taken in, then sent.
// Activations a conv already ReLU'd pass as they are (SHIFT 0, see relu_in).
template<int MAP_HEIGHT, int MAP_WIDTH, int SHIFT, int CHANNELS, typename IN_T>
void relu_pool_up_patch(hls::stream<vec_t<IN_T, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out) {
    act_t block_max[MAP_WIDTH / 2][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=block_max type=complete dim=2
    act_t left[CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    pairs: for (int j = 0; j < MAP_HEIGHT / 2; j++) {
        take: for (int i = 0; i < 2 * MAP_WIDTH; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=block_max type=inter false
            const int x = i % MAP_WIDTH;
            const vec_t<IN_T, CHANNELS> value = in.read();
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                const act_t a = relu_in<SHIFT>(value.v[ch]);
                if (x % 2 == 0) {
                    left[ch] = a;
                } else {
                    const act_t pair = max2(left[ch], a);
                    block_max[x / 2][ch] = (i < MAP_WIDTH) ? pair : max2(block_max[x / 2][ch], pair);
                }
            }
        }
        give: for (int i = 0; i < 2 * MAP_WIDTH; i++) {
            #pragma HLS PIPELINE II=1
            vec_t<act_t, CHANNELS> pooled;
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                pooled.v[ch] = block_max[(i % MAP_WIDTH) / 2][ch];
            }
            out.write(pooled);
        }
    }
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int SHIFT, int CHANNELS, typename IN_T>
void relu_pool_up(hls::stream<vec_t<IN_T, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out,
                  int height, int width) {
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_pool_up_patch<MAP_HEIGHT, MAP_WIDTH, SHIFT>(in, out);
    }
}
