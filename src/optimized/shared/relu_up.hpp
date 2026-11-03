#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// ReLU, round off ROUND_BITS bits, upsample by 2: each row is taken in, then sent
// twice, every pixel twice. Activations a conv already ReLU'd pass as they are
// (see to_act).
template<int MAP_HEIGHT, int MAP_WIDTH, int ROUND_BITS, int CHANNELS, typename IN_T>
void relu_up_patch(hls::stream<vec_t<IN_T, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out) {
    act_t line[MAP_WIDTH][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=line type=complete dim=2
    rows: for (int y = 0; y < MAP_HEIGHT; y++) {
        take: for (int x = 0; x < MAP_WIDTH; x++) {
            #pragma HLS PIPELINE II=1
            const vec_t<IN_T, CHANNELS> value = in.read();
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                line[x][ch] = to_act<ROUND_BITS>(value.v[ch]);
            }
        }
        give: for (int i = 0; i < 4 * MAP_WIDTH; i++) {
            #pragma HLS PIPELINE II=1
            vec_t<act_t, CHANNELS> upsampled;
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                upsampled.v[ch] = line[(i % (2 * MAP_WIDTH)) / 2][ch];
            }
            out.write(upsampled);
        }
    }
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int ROUND_BITS, int CHANNELS, typename IN_T>
void relu_up(hls::stream<vec_t<IN_T, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out,
             int height, int width) {
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_up_patch<MAP_HEIGHT, MAP_WIDTH, ROUND_BITS>(in, out);
    }
}
