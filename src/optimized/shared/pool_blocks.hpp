#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// 2x2 max pooling of a map in 2x2-block order (wino_conv3x3.hpp): a block is
// one pooling window. Its activations are rounded already, which is monotonic,
// so pooling them is pooling the sums.

// One pooled pixel per block, in raster order.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int CHANNELS>
void pool_blocks(hls::stream<vec_t<act_t, 4 * CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out,
                 int height, int width) {
    constexpr int BLOCKS = (MAP_HEIGHT / 2) * (MAP_WIDTH / 2), MAX_BLOCKS = 665 * BLOCKS;
    (void)MAX_BLOCKS;   // only the pragma uses it
    const int blocks = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * BLOCKS;
    blocks: for (int k = 0; k < blocks; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=BLOCKS max=MAX_BLOCKS
        const vec_t<act_t, 4 * CHANNELS> block = in.read();
        vec_t<act_t, CHANNELS> pooled;
        for (int ch = 0; ch < CHANNELS; ch++) {
            #pragma HLS UNROLL
            pooled.v[ch] = max2(max2(block.v[ch], block.v[CHANNELS + ch]),
                                max2(block.v[2 * CHANNELS + ch], block.v[3 * CHANNELS + ch]));
        }
        out.write(pooled);
    }
}

// Every block also goes to skip, and the pooled map leaves in block order of its
// own: a pooled block comes from four blocks in two block rows, so the even
// row's are kept until the odd row's arrive.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int CHANNELS>
void pool_skip_blocks(hls::stream<vec_t<act_t, 4 * CHANNELS> >& in, hls::stream<vec_t<act_t, 4 * CHANNELS> >& skip,
                      hls::stream<vec_t<act_t, 4 * CHANNELS> >& out, int height, int width) {
    constexpr int BLOCKS_HIGH = MAP_HEIGHT / 2, BLOCKS_WIDE = MAP_WIDTH / 2;
    constexpr int BLOCKS = BLOCKS_HIGH * BLOCKS_WIDE, MAX_BLOCKS = 665 * BLOCKS;
    (void)MAX_BLOCKS;   // only the pragma uses it
    act_t even_left[BLOCKS_WIDE / 2][CHANNELS], even_right[BLOCKS_WIDE / 2][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=even_left type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=even_right type=complete dim=2
    act_t odd_left[CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=odd_left type=complete
    const int blocks = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * BLOCKS;
    int block_row = 0, block_col = 0;
    blocks: for (int k = 0; k < blocks; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=BLOCKS max=MAX_BLOCKS
        #pragma HLS DEPENDENCE variable=even_left type=inter false
        #pragma HLS DEPENDENCE variable=even_right type=inter false
        const vec_t<act_t, 4 * CHANNELS> block = in.read();
        skip.write(block);
        vec_t<act_t, 4 * CHANNELS> pooled_block;
        for (int ch = 0; ch < CHANNELS; ch++) {
            #pragma HLS UNROLL
            const act_t pooled = max2(max2(block.v[ch], block.v[CHANNELS + ch]),
                                      max2(block.v[2 * CHANNELS + ch], block.v[3 * CHANNELS + ch]));
            if (block_row % 2 == 0) {
                if (block_col % 2 == 0) even_left[block_col / 2][ch] = pooled;
                else even_right[block_col / 2][ch] = pooled;
            } else if (block_col % 2 == 0) {
                odd_left[ch] = pooled;
            } else {
                pooled_block.v[ch] = even_left[block_col / 2][ch];
                pooled_block.v[CHANNELS + ch] = even_right[block_col / 2][ch];
                pooled_block.v[2 * CHANNELS + ch] = odd_left[ch];
                pooled_block.v[3 * CHANNELS + ch] = pooled;
            }
        }
        if (block_row % 2 == 1 && block_col % 2 == 1) out.write(pooled_block);
        if (block_col < BLOCKS_WIDE - 1) block_col++;
        else { block_col = 0; block_row = block_row < BLOCKS_HIGH - 1 ? block_row + 1 : 0; }
    }
}
