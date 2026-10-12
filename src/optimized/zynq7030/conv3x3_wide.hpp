#pragma once

#include "../shared/conv3x3_stream.hpp"
#include "../shared/patch_grid.hpp"

// conv3x3_wide for the Zynq-7030: the same layer as zcu104/conv3x3_wide.hpp
// (P input channels x every output channel per cycle, chunk by chunk, int8
// input), with one multiply per DSP. The 7-series DSP48E1 multiplies 25 x 18
// bits, too narrow for two weights and their products (see
// zcu104/conv3x3_packed.hpp). OUT x P x 9 DSPs, IN_PAD / P x H x W + W + 1
// cycles per patch. The input channels are padded with zero weights up to
// IN_PAD, a multiple of P: the producer sends IN_PAD.
//
// The image is cut into PH x PW patches (patch_grid.hpp); the layer is H x W
// of each. WEIGHT and BIAS: the layer's quantized int8 arrays, template
// arguments so that they stay a constant ROM inside this process. BIAS_SHIFT:
// the sums' fraction bits minus the bias's. Out: every pixel's OUT sums,
// unscaled, raster order.
template<int PH, int PW, int H, int W, int IN, int OUT, int P, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT]>
void conv3x3_wide(hls::stream<vec_t<act8_t, P> >& in, hls::stream<vec_t<acc_t, OUT> >& out,
                  int height, int width) {
    constexpr int IN_PAD = (IN + P - 1) / P * P;
    wgt_t wl[OUT][IN_PAD][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wl type=cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[OUT];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete

    load_conv_weights<OUT, IN, IN_PAD>(WEIGHT, BIAS, BIAS_SHIFT, wl, bl);
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_stream<H, W, IN_PAD, OUT, P, 1>(in, wl, bl, out);
    }
}
