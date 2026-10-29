#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "pixel_io.hpp"
#include "conv1d_engine.hpp"

// A 1x1 conv of a per-pixel network on uint8 activations: every input channel
// at once, OUT_PER_CYCLE output channels a cycle. The outputs are shifted into a
// register array, so every index is a constant. RELU: outputs ReLU'd, rounded
// off SHIFT bits, saturated at ACT_MAX; otherwise the head's scores, the sums
// shifted left by SHIFT. LANE: which pixels (pixel_lane).
template<bool RELU, int SHIFT, int ACT_MAX> struct pointwise1d_out {
    typedef uact_t type;
    static type of(acc_t sum) {
        #pragma HLS INLINE
        return requant_relu8<SHIFT, ACT_MAX>(sum);
    }
};
template<int SHIFT, int ACT_MAX> struct pointwise1d_out<false, SHIFT, ACT_MAX> {
    typedef acc_t type;
    static type of(acc_t sum) {
        #pragma HLS INLINE
        return sum * ((acc_t)1 << SHIFT);
    }
};

template<int IN_CHANNELS, int OUT_CHANNELS, int OUT_PER_CYCLE, bool RELU, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][1][1], const wgt_t (&BIAS)[OUT_CHANNELS], bool DSP,
         typename LANE>
void pointwise1d(hls::stream<vec_t<uact_t, IN_CHANNELS> >& in,
                 hls::stream<vec_t<typename pointwise1d_out<RELU, SHIFT, ACT_MAX>::type, OUT_CHANNELS> >& out,
                 int height, int width) {
    typedef typename pointwise1d_out<RELU, SHIFT, ACT_MAX>::type out_t;
    constexpr int STEPS = OUT_CHANNELS / OUT_PER_CYCLE;
    wgt_t weight[STEPS][OUT_PER_CYCLE][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    acc_t bias[STEPS][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    load: for (int o = 0; o < OUT_CHANNELS; o++) {
        for (int i = 0; i < IN_CHANNELS; i++) {
            #pragma HLS PIPELINE II=1
            weight[o / OUT_PER_CYCLE][o % OUT_PER_CYCLE][i] = WEIGHT[o][i][0][0];
        }
        bias[o / OUT_PER_CYCLE][o % OUT_PER_CYCLE] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }

    const uint32_t pixels = LANE::count(height, width);
    uact_t pixel[IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=pixel type=complete
    out_t outputs[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=outputs type=complete
    int step = 0;
    px: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=7836192
        if (step == 0) {
            const vec_t<uact_t, IN_CHANNELS> value = in.read();
            for (int i = 0; i < IN_CHANNELS; i++) {
                #pragma HLS UNROLL
                pixel[i] = value.v[i];
            }
        }
        for (int k = 0; k < OUT_CHANNELS - OUT_PER_CYCLE; k++) {
            #pragma HLS UNROLL
            outputs[k] = outputs[k + OUT_PER_CYCLE];
        }
        for (int j = 0; j < OUT_PER_CYCLE; j++) {
            #pragma HLS UNROLL
            acc_t sum = bias[step][j];
            for (int i = 0; i < IN_CHANNELS; i++) {
                #pragma HLS UNROLL
                sum += mul<DSP, acc_t>(pixel[i], weight[step][j][i]);
            }
            outputs[OUT_CHANNELS - OUT_PER_CYCLE + j] = pointwise1d_out<RELU, SHIFT, ACT_MAX>::of(sum);
        }
        if (step == STEPS - 1) {
            vec_t<out_t, OUT_CHANNELS> result;
            for (int i = 0; i < OUT_CHANNELS; i++) {
                #pragma HLS UNROLL
                result.v[i] = outputs[i];
            }
            out.write(result);
            step = 0;
        } else {
            step++;
        }
    }
}
