#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// relu_pool_up: ReLU, 2x2 max-pool, then 2x nearest upsample, H x W -> H x W
// per patch: every 2x2 block replaced by its maximum. Reads a conv layer's
// sums (all channels of a pixel, raster order), with S fraction bits, and
// writes activations with S - SHIFT the same way. Each pair of rows is taken
// in, then given out. The image is cut into PH x PW patches (patch_grid.hpp);
// H x W is the layer's size of a patch.
template<int H, int W, int SHIFT, int CH>
void relu_pool_up_patch(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out) {
    act_t blk[W / 2][CH];   // the maxima of this row pair's 2x2 blocks
    #pragma HLS ARRAY_PARTITION variable=blk type=complete dim=2
    act_t left[CH];
    #pragma HLS ARRAY_PARTITION variable=left type=complete
    pairs: for (int j = 0; j < H / 2; j++) {
        take: for (int i = 0; i < 2 * W; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=blk type=inter false
            const int x = i % W;
            const vec_t<acc_t, CH> v = in.read();
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                const act_t a = requant_relu<SHIFT>(v.v[ch]);
                if (x % 2 == 0) {
                    left[ch] = a;
                } else {
                    const act_t m = max2(left[ch], a);
                    blk[x / 2][ch] = (i < W) ? m : max2(blk[x / 2][ch], m);
                }
            }
        }
        give: for (int i = 0; i < 2 * W; i++) {
            #pragma HLS PIPELINE II=1
            vec_t<act_t, CH> o;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                o.v[ch] = blk[(i % W) / 2][ch];
            }
            out.write(o);
        }
    }
}

template<int PH, int PW, int H, int W, int SHIFT, int CH>
void relu_pool_up(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out,
                  int height, int width) {
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_pool_up_patch<H, W, SHIFT>(in, out);
    }
}
