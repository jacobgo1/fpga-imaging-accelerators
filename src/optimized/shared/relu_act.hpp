#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// ReLU and round off SHIFT bits: a conv's sums to the next conv's activations,
// one pixel a cycle. PIXELS is the layer's pixels in a patch.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS, int SHIFT, int CHANNELS>
void relu_act(hls::stream<vec_t<acc_t, CHANNELS> >& in, hls::stream<vec_t<act_t, CHANNELS> >& out,
              int height, int width) {
    constexpr int MAX_PIXELS = 665 * PIXELS;
    (void)MAX_PIXELS;   // only the pragma uses it
    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PIXELS;
    px: for (int k = 0; k < pixels; k++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=PIXELS max=MAX_PIXELS
        const vec_t<acc_t, CHANNELS> sums = in.read();
        vec_t<act_t, CHANNELS> activations;
        for (int ch = 0; ch < CHANNELS; ch++) {
            #pragma HLS UNROLL
            activations.v[ch] = requant_relu<SHIFT>(sums.v[ch]);
        }
        out.write(activations);
    }
}
