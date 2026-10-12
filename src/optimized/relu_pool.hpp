#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// relu_pool: ReLU, then 2x2 max-pool, H x W -> H/2 x W/2 per patch. Reads a
// conv layer's sums (all channels of a pixel, raster order), with S fraction
// bits, and writes activations with S - SHIFT the same way: the order the next
// conv3x3_narrow reads. A pooled pixel leaves as soon as its block's last
// value (odd row, odd column) arrives. The image is cut into PH x PW patches
// (patch_grid.hpp); H x W is the layer's size of a patch.
template<int H, int W, int SHIFT, int CH>
void relu_pool_patch(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out) {
    act_t row[W / 2][CH];   // the even row's pairs, pooled
    #pragma HLS ARRAY_PARTITION variable=row type=complete dim=2
    act_t left[CH];         // the even column's value
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=row type=inter false
            const vec_t<acc_t, CH> v = in.read();
            vec_t<act_t, CH> o;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                const act_t a = requant_relu<SHIFT>(v.v[ch]);
                if (x % 2 == 0) {
                    left[ch] = a;
                } else {
                    const act_t m = max2(left[ch], a);
                    if (y % 2 == 0) row[x / 2][ch] = m;
                    else o.v[ch] = max2(row[x / 2][ch], m);
                }
            }
            if (y % 2 == 1 && x % 2 == 1) out.write(o);
        }
}

template<int PH, int PW, int H, int W, int SHIFT, int CH>
void relu_pool(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out,
               int height, int width) {
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_pool_patch<H, W, SHIFT>(in, out);
    }
}
