#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A map in 2x2-block order (wino_conv3x3.hpp) back to raster order, one pixel a
// cycle. A block row is two pixel rows: the top one goes out as the blocks come
// in, the bottom one is kept and goes out after it.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int CHANNELS, typename T>
void blocks_to_rows(hls::stream<vec_t<T, 4 * CHANNELS> >& in, hls::stream<vec_t<T, CHANNELS> >& out,
                    int height, int width) {
    constexpr int BLOCKS_WIDE = MAP_WIDTH / 2, STEPS_PER_BLOCK_ROW = 4 * BLOCKS_WIDE;
    constexpr int STEPS = (MAP_HEIGHT / 2) * STEPS_PER_BLOCK_ROW, MAX_STEPS = 665 * STEPS;
    (void)MAX_STEPS;   // only the pragma uses it
    T top_right[CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=top_right type=complete
    T bottom_left[BLOCKS_WIDE][CHANNELS], bottom_right[BLOCKS_WIDE][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bottom_left type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bottom_right type=complete dim=2
    const int steps = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * STEPS;
    int step = 0;
    steps: for (int k = 0; k < steps; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=STEPS max=MAX_STEPS
        #pragma HLS DEPENDENCE variable=bottom_left type=inter false
        #pragma HLS DEPENDENCE variable=bottom_right type=inter false
        vec_t<T, CHANNELS> pixel;
        if (step < 2 * BLOCKS_WIDE && step % 2 == 0) {
            const vec_t<T, 4 * CHANNELS> block = in.read();
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                pixel.v[ch] = block.v[ch];
                top_right[ch] = block.v[CHANNELS + ch];
                bottom_left[step / 2][ch] = block.v[2 * CHANNELS + ch];
                bottom_right[step / 2][ch] = block.v[3 * CHANNELS + ch];
            }
        } else {
            const int x = step < 2 * BLOCKS_WIDE ? 0 : (step - 2 * BLOCKS_WIDE) / 2;
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                pixel.v[ch] = step < 2 * BLOCKS_WIDE ? top_right[ch]
                            : step % 2 == 0 ? bottom_left[x][ch] : bottom_right[x][ch];
            }
        }
        out.write(pixel);
        step = step < STEPS_PER_BLOCK_ROW - 1 ? step + 1 : 0;
    }
}
