#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// relu_act: ReLU, then round off SHIFT bits to int16 activations: a conv
// layer's sums (all channels of a pixel, raster order) into the next conv's
// input, the same order. One pixel a cycle. The image is cut into PH x PW
// patches (patch_grid.hpp); PIX is the layer's pixels in a patch.
template<int PH, int PW, int PIX, int SHIFT, int CH>
void relu_act(hls::stream<vec_t<acc_t, CH> >& in, hls::stream<vec_t<act_t, CH> >& out,
              int height, int width) {
    constexpr int MAX_PIX = 665 * PIX;
    (void)MAX_PIX;   // only the pragma uses it
    const int n = patch_count<PH, PW>(height, width) * PIX;
    px: for (int k = 0; k < n; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=PIX max=MAX_PIX
        const vec_t<acc_t, CH> v = in.read();
        vec_t<act_t, CH> o;
        for (int c = 0; c < CH; c++) {
            #pragma HLS UNROLL
            o.v[c] = requant_relu<SHIFT>(v.v[c]);
        }
        out.write(o);
    }
}
