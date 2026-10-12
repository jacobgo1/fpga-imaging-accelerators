#pragma once

#include "../shared/conv3x3_packed.hpp"

// conv3x3_wide for the Zynq-7020: a 3x3 conv layer for many input channels (the
// bands, in the first layer), P input channels x every output channel per
// cycle, two multiplies per DSP (shared/conv3x3_packed.hpp, which has the
// rest). The 7-series DSP48E1 multiplies 25 x 18 bits, two fewer than the
// ZCU104's: the two weights go 16 bits apart, and a group's low sum then holds
// only 2 products (2 x 16256 < 2^15) before it is split. The splits and their
// sums are LUTs; the DSPs only multiply and add pairs.
template<int PH, int PW, int H, int W, int IN, int OUT, int P, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT]>
void conv3x3_wide(hls::stream<vec_t<act8_t, P> >& in, hls::stream<vec_t<acc_t, OUT> >& out,
                  int height, int width) {
    #pragma HLS INLINE
    conv3x3_packed_layer<PH, PW, H, W, IN, OUT, P, BIAS_SHIFT, WEIGHT, BIAS, 16, 2>(in, out, height, width);
}
