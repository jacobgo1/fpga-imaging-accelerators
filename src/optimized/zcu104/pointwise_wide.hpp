#pragma once

#include "../shared/pointwise_packed.hpp"

// pointwise_wide for the ZCU104: a 1x1 conv layer on the int8 bands, P bands x
// every output channel a cycle, two multiplies per DSP
// (shared/pointwise_packed.hpp, which has the rest). The UltraScale+ DSP48E2
// multiplies 27 x 18 bits: room for two weights 18 bits apart, and for groups
// of 8 products before they are split (as zcu104/conv3x3_wide.hpp).
template<int PH, int PW, int PIX, int IN, int OUT, int P, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][1][1], const wgt_t (&BIAS)[OUT]>
void pointwise_wide(hls::stream<vec_t<act8_t, P> >& in, hls::stream<vec_t<act_t, OUT> >& out,
                    int height, int width) {
    #pragma HLS INLINE
    pointwise_packed<PH, PW, PIX, IN, OUT, P, BIAS_SHIFT, SHIFT, WEIGHT, BIAS, 18, 8>(in, out, height, width);
}
