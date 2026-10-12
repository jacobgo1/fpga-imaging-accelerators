#pragma once

#include "../shared/conv3x3_packed.hpp"

// conv3x3_wide for the ZCU104: a 3x3 conv layer for many input channels (the
// bands, in the first layer), P input channels x every output channel per
// cycle, two multiplies per DSP (shared/conv3x3_packed.hpp, which has the
// rest). The UltraScale+ DSP48E2 multiplies 27 x 18 bits: room for two weights
// 18 bits apart, and for groups of 8 products before they are split.
template<int PH, int PW, int H, int W, int IN, int OUT, int P, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT]>
void conv3x3_wide(hls::stream<vec_t<act8_t, P> >& in, hls::stream<vec_t<acc_t, OUT> >& out,
                  int height, int width) {
    #pragma HLS INLINE
    conv3x3_packed_layer<PH, PW, H, W, IN, OUT, P, BIAS_SHIFT, WEIGHT, BIAS, 18, 8>(in, out, height, width);
}
