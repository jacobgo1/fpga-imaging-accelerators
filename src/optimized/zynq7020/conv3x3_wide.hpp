#pragma once

#include "../shared/conv3x3_packed.hpp"

// The 7-series DSP48E1 takes 25 bits: two weights 16 bits apart, groups of 2 products.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS]>
void conv3x3_wide(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in, hls::stream<vec_t<acc_t, OUT_CHANNELS> >& out,
                  int height, int width) {
    #pragma HLS INLINE
    conv3x3_packed_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
                         BIAS_SHIFT, WEIGHT, BIAS, 16, 2>(in, out, height, width);
}
