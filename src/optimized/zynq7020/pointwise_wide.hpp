#pragma once

#include "../shared/pointwise_packed.hpp"

// pointwise_wide for the Zynq-7020: a 1x1 conv layer on the int8 bands, P
// bands x OUT / OPASS output channels a cycle, two multiplies per DSP
// (shared/pointwise_packed.hpp, which has the rest). The 7-series DSP48E1
// multiplies 25 x 18 bits, two fewer than the ZCU104's: the two weights go 16
// bits apart, and a group holds only 2 products before it is split (as
// zynq7020/conv3x3_wide.hpp).
template<int PH, int PW, int PIX, int IN, int OUT, int P, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][1][1], const wgt_t (&BIAS)[OUT], int OPASS = 1>
void pointwise_wide(hls::stream<vec_t<act8_t, P> >& in, hls::stream<vec_t<act_t, OUT> >& out,
                    int height, int width) {
    #pragma HLS INLINE
    pointwise_packed<PH, PW, PIX, IN, OUT, P, BIAS_SHIFT, SHIFT, WEIGHT, BIAS, 16, 2, OPASS>(in, out, height, width);
}
