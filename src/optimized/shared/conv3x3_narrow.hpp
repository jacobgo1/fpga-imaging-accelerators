#pragma once

#include "conv3x3_stream.hpp"
#include "patch_grid.hpp"

// conv3x3_narrow: a 3x3 conv layer with few channels and time to spare: every
// input channel of a pixel at once, one output channel per cycle (conv3x3_stream
// with P = IN, STEPS = OUT). IN x 9 multipliers, OUT cycles per pixel. The
// weights: one small memory per (input channel, tap), addressed by the output
// channel.
//
// The image is cut into PH x PW patches (patch_grid.hpp); the layer is H x W
// of each. WEIGHT and BIAS: the layer's quantized int8 arrays, template
// arguments so that they stay a constant ROM inside this process. BIAS_SHIFT:
// the sums' fraction bits minus the bias's. DSP: the multipliers in DSPs, or
// in LUTs. Out: every pixel's OUT sums, unscaled, raster order.
template<int PH, int PW, int H, int W, int IN, int OUT, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT], bool DSP>
void conv3x3_narrow(hls::stream<vec_t<act_t, IN> >& in, hls::stream<vec_t<acc_t, OUT> >& out,
                    int height, int width) {
    wgt_t wl[OUT][IN][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[OUT];

    load_conv_weights<OUT, IN, BIAS_SHIFT>(WEIGHT, BIAS, wl, bl);
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_stream<H, W, IN, OUT, IN, OUT, DSP>(in, wl, bl, out);
    }
}

