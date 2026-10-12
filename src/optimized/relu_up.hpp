#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// relu_up: ReLU, then 2x nearest upsample, H x W -> 2H x 2W per patch. Reads a
// conv layer's sums (all channels of a pixel, raster order), with S fraction
// bits, and writes activations with S - SHIFT the same way. Each row is taken
// in, then given out twice, every pixel twice. The image is cut into PH x PW
// patches (patch_grid.hpp); H x W is the layer's input size of a patch.
template<int H, int W, int SHIFT, int CH>
void relu_up_patch(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out) {
    act_t line[W][CH];
    #pragma HLS ARRAY_PARTITION variable=line type=complete dim=2
    rows: for (int y = 0; y < H; y++) {
        take: for (int x = 0; x < W; x++) {
            #pragma HLS PIPELINE II=1
            const vec_t<acc_t, CH> v = in.read();
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                line[x][ch] = requant_relu<SHIFT>(v.v[ch]);
            }
        }
        give: for (int i = 0; i < 4 * W; i++) {
            #pragma HLS PIPELINE II=1
            vec_t<act_t, CH> o;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                o.v[ch] = line[(i % (2 * W)) / 2][ch];
            }
            out.write(o);
        }
    }
}

template<int PH, int PW, int H, int W, int SHIFT, int CH>
void relu_up(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out,
             int height, int width) {
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_up_patch<H, W, SHIFT>(in, out);
    }
}
