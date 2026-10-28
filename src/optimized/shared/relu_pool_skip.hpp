#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// relu_pool_skip: relu_pool (ReLU, 2x2 max-pool, H x W -> H/2 x W/2 per patch)
// that also sends every activation on to skip, as it is: a U-Net encoder
// block's output, both to the next block and to the skip connection. In: a
// conv layer's sums with S fraction bits, all channels of a pixel, raster
// order; out: activations with S - SHIFT, the same order. The image is cut
// into PH x PW patches (patch_grid.hpp); H x W is the layer's size of a patch.
// The input can also be activations a conv already ReLU'd and rounded
// (conv3x3_grouped's RELU_SHIFT): then SHIFT is 0 and this only moves them
// (relu_in, fixed_point.hpp).
template<int H, int W, int SHIFT, int CH, typename IN_T>
void relu_pool_skip_patch(hls::stream<vec_t<IN_T, CH> >& in, hls::stream<vec_t<act_t, CH> >& skip,
                          hls::stream<vec_t<act_t, CH> >& out) {
    act_t row[W / 2][CH];   // the even row's pairs, pooled
    #pragma HLS ARRAY_PARTITION variable=row type=complete dim=2
    act_t left[CH];         // the even column's value
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=row type=inter false
            const vec_t<IN_T, CH> v = in.read();
            vec_t<act_t, CH> a, o;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                a.v[ch] = relu_in<SHIFT>(v.v[ch]);
                if (x % 2 == 0) {
                    left[ch] = a.v[ch];
                } else {
                    const act_t m = max2(left[ch], a.v[ch]);
                    if (y % 2 == 0) row[x / 2][ch] = m;
                    else o.v[ch] = max2(row[x / 2][ch], m);
                }
            }
            skip.write(a);
            if (y % 2 == 1 && x % 2 == 1) out.write(o);
        }
}

template<int PH, int PW, int H, int W, int SHIFT, int CH, typename IN_T>
void relu_pool_skip(hls::stream<vec_t<IN_T, CH> >& in, hls::stream<vec_t<act_t, CH> >& skip,
                    hls::stream<vec_t<act_t, CH> >& out, int height, int width) {
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_pool_skip_patch<H, W, SHIFT>(in, skip, out);
    }
}
