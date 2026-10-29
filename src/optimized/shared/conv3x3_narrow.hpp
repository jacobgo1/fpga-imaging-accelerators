#pragma once

#include "conv3x3_stream.hpp"
#include "patch_grid.hpp"

// A 3x3 conv with few channels: every input channel at once, OUT_CHANNELS / STEPS
// output channels a cycle. The weights stay a ROM inside this process.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int BIAS_SHIFT, const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS],
         bool DSP, int STEPS = OUT_CHANNELS>
void conv3x3_narrow(hls::stream<vec_t<act_t, IN_CHANNELS> >& in, hls::stream<vec_t<acc_t, OUT_CHANNELS> >& out,
                    int height, int width) {
    constexpr int OUT_PER_STEP = OUT_CHANNELS / STEPS;
    (void)OUT_PER_STEP;   // only the pragmas use it
    wgt_t weight[OUT_CHANNELS][IN_CHANNELS][3][3];
    #pragma HLS ARRAY_PARTITION variable=weight type=cyclic factor=OUT_PER_STEP dim=1
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=4
    acc_t bias[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bias type=cyclic factor=OUT_PER_STEP dim=1

    load_conv_weights<OUT_CHANNELS, IN_CHANNELS, BIAS_SHIFT>(WEIGHT, BIAS, weight, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_stream<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, STEPS, DSP>(in, weight, bias, out);
    }
}
