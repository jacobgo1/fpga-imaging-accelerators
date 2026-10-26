#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// 2x2 max-pooling of a map in 2x2-block order (wino_conv3x3.hpp): a block is
// exactly one pooling window, so each block becomes one pixel, and the pooled
// map comes out in raster order. The activations are ReLU'd and rounded
// already: rounding is monotonic, so pooling them is pooling the sums.
//
// The image is cut into PH x PW patches (patch_grid.hpp); H x W is the map's
// size in a patch (pixels), H/2 x W/2 its blocks.

// pool_blocks: one pooled pixel per block, raster order (for wino_up_conv3x3).
template<int PH, int PW, int H, int W, int CH>
void pool_blocks(hls::stream<vec_t<act_t, 4 * CH> >& in, hls::stream<vec_t<act_t, CH> >& out,
                 int height, int width) {
    constexpr int BLOCKS = (H / 2) * (W / 2), MAX_BLOCKS = 665 * BLOCKS;
    (void)BLOCKS; (void)MAX_BLOCKS;   // only the pragma uses them
    const int n = patch_count<PH, PW>(height, width) * BLOCKS;
    blocks: for (int k = 0; k < n; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=BLOCKS max=MAX_BLOCKS
        const vec_t<act_t, 4 * CH> v = in.read();
        vec_t<act_t, CH> o;
        for (int c = 0; c < CH; c++) {
            #pragma HLS UNROLL
            o.v[c] = max2(max2(v.v[c], v.v[CH + c]), max2(v.v[2 * CH + c], v.v[3 * CH + c]));
        }
        out.write(o);
    }
}

// pool_skip_blocks: the encoder's first block, both ways. Every block goes on
// to skip as it is (the U-Net's skip connection), and the pooled map goes out
// in 2x2-block order of its own (for the next wino_conv3x3): a pooled block
// is the pooled pixels of four blocks, two in each of a pair of block rows, so
// the even row's are kept until the odd row's arrive.
template<int PH, int PW, int H, int W, int CH>
void pool_skip_blocks(hls::stream<vec_t<act_t, 4 * CH> >& in, hls::stream<vec_t<act_t, 4 * CH> >& skip,
                      hls::stream<vec_t<act_t, 4 * CH> >& out, int height, int width) {
    constexpr int GH = H / 2, GW = W / 2, BLOCKS = GH * GW, MAX_BLOCKS = 665 * BLOCKS;
    (void)BLOCKS; (void)MAX_BLOCKS;   // only the pragma uses them
    static_assert(GH % 2 == 0 && GW % 2 == 0, "the pooled map must be whole 2x2 blocks too");
    act_t even_l[GW / 2][CH], even_r[GW / 2][CH];   // the even block row's pooled pixels, by column parity
    #pragma HLS ARRAY_PARTITION variable=even_l type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=even_r type=complete dim=2
    act_t left[CH];                                 // the odd row's even column
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    const int n = patch_count<PH, PW>(height, width) * BLOCKS;
    int by = 0, bx = 0;
    blocks: for (int k = 0; k < n; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=BLOCKS max=MAX_BLOCKS
        #pragma HLS DEPENDENCE variable=even_l type=inter false
        #pragma HLS DEPENDENCE variable=even_r type=inter false
        const vec_t<act_t, 4 * CH> v = in.read();
        skip.write(v);
        vec_t<act_t, 4 * CH> o;
        for (int c = 0; c < CH; c++) {
            #pragma HLS UNROLL
            const act_t m = max2(max2(v.v[c], v.v[CH + c]), max2(v.v[2 * CH + c], v.v[3 * CH + c]));
            if (by % 2 == 0) {
                if (bx % 2 == 0) even_l[bx / 2][c] = m;
                else even_r[bx / 2][c] = m;
            } else if (bx % 2 == 0) {
                left[c] = m;
            } else {
                o.v[c] = even_l[bx / 2][c];
                o.v[CH + c] = even_r[bx / 2][c];
                o.v[2 * CH + c] = left[c];
                o.v[3 * CH + c] = m;
            }
        }
        if (by % 2 == 1 && bx % 2 == 1) out.write(o);
        if (bx < GW - 1) bx++;
        else { bx = 0; by = by < GH - 1 ? by + 1 : 0; }
    }
}
