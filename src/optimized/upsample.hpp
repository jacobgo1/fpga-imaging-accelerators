#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// Upsample by 2 (nearest): each row is taken in, then sent twice, every pixel twice.
// T: the activations' type, int16 or uint8.
template<int MAP_HEIGHT, int MAP_WIDTH, int CHANNELS, typename T = act_t>
void upsample_patch(hls::stream<vec_t<T, CHANNELS> >& in, hls::stream<vec_t<T, CHANNELS> >& out) {
    T line[MAP_WIDTH][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=line type=complete dim=2
    rows: for (int y = 0; y < MAP_HEIGHT; y++) {
        take: for (int x = 0; x < MAP_WIDTH; x++) {
            #pragma HLS PIPELINE II=1
            const vec_t<T, CHANNELS> value = in.read();
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                line[x][ch] = value.v[ch];
            }
        }
        give: for (int i = 0; i < 4 * MAP_WIDTH; i++) {
            #pragma HLS PIPELINE II=1
            vec_t<T, CHANNELS> upsampled;
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                upsampled.v[ch] = line[(i % (2 * MAP_WIDTH)) / 2][ch];
            }
            out.write(upsampled);
        }
    }
}

// MAP_HEIGHT x MAP_WIDTH in, twice that out.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int CHANNELS, typename T = act_t>
void upsample(hls::stream<vec_t<T, CHANNELS> >& in, hls::stream<vec_t<T, CHANNELS> >& out,
              int height, int width) {
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        upsample_patch<MAP_HEIGHT, MAP_WIDTH>(in, out);
    }
}
