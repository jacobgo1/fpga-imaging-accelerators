#pragma once

#include "../shared/pointwise_packed.hpp"

// The 7-series DSP48E1 takes 25 bits: two weights 16 bits apart, groups of 2 products.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE,
         int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][1][1], const wgt_t (&BIAS)[OUT_CHANNELS],
         int OUTPUT_PASSES = 1>
void pointwise_wide(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in, hls::stream<vec_t<act_t, OUT_CHANNELS> >& out,
                    int height, int width) {
    #pragma HLS INLINE
    pointwise_packed<PATCH_HEIGHT, PATCH_WIDTH, PIXELS, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, BIAS_SHIFT, SHIFT,
                     WEIGHT, BIAS, 16, 2, OUTPUT_PASSES>(in, out, height, width);
}
