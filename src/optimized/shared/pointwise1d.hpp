#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "pixel_io.hpp"        // pixel_lane
#include "conv1d_engine.hpp"   // uact_t, requant_relu8

// pointwise1d: a 1x1 conv layer of a per-pixel network on its uint8
// activations, one pixel after another: every input channel of a pixel at once,
// PO output channels a cycle, OUT / PO cycles a pixel, IN x PO multipliers
// (DSP: in DSPs, or LUTs). The weights: one small memory per (output lane,
// input channel), OUT / PO deep, addressed by the step. The outputs are shifted
// in at the end of a register array, so every index is a constant.
//
// RELU: each output ReLU'd, rounded off SHIFT bits and saturated at ACT_MAX, a
// uint8 activation for the next layer (out: uact_t). Otherwise the network's
// head: the plain sums, bias included, times 2^SHIFT (out: acc_t, the scores at
// their final fraction bits; SHIFT then shifts left).
//
// WEIGHT and BIAS: the layer's int8 arrays, Conv2d's [OUT][IN][1][1], template
// arguments so that they stay a ROM inside this process. BIAS_SHIFT: the sums'
// fraction bits minus the bias's. LANE: which pixels (pixel_lane).
template<bool RELU, int SHIFT, int ACT_MAX> struct pointwise1d_out {
    typedef uact_t type;
    static type of(acc_t v) {
        #pragma HLS INLINE
        return requant_relu8<SHIFT, ACT_MAX>(v);
    }
};
template<int SHIFT, int ACT_MAX> struct pointwise1d_out<false, SHIFT, ACT_MAX> {
    typedef acc_t type;
    static type of(acc_t v) {
        #pragma HLS INLINE
        static_assert(SHIFT >= 0, "the scores keep every fraction bit of the sums");
        return v * ((acc_t)1 << SHIFT);
    }
};

template<int IN, int OUT, int PO, bool RELU, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OUT][IN][1][1], const wgt_t (&BIAS)[OUT], bool DSP, typename LANE>
void pointwise1d(hls::stream<vec_t<uact_t, IN> >& in,
                 hls::stream<vec_t<typename pointwise1d_out<RELU, SHIFT, ACT_MAX>::type, OUT> >& out,
                 int height, int width) {
    static_assert(OUT % PO == 0, "PO must divide OUT");
    static_assert(BIAS_SHIFT >= 0, "the bias must not have more fraction bits than the sums");
    typedef typename pointwise1d_out<RELU, SHIFT, ACT_MAX>::type out_t;
    constexpr int STEPS = OUT / PO;
    wgt_t wl[STEPS][PO][IN];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    acc_t bl[STEPS][PO];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete dim=2
    load: for (int oc = 0; oc < OUT; oc++) {
        for (int ic = 0; ic < IN; ic++) {
            #pragma HLS PIPELINE II=1
            wl[oc / PO][oc % PO][ic] = WEIGHT[oc][ic][0][0];
        }
        bl[oc / PO][oc % PO] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
    }

    const uint32_t pixels = LANE::count(height, width);
    uact_t x[IN];               // this pixel
    #pragma HLS ARRAY_PARTITION variable=x type=complete
    out_t ob[OUT];              // its outputs, PO shifted in a step
    #pragma HLS ARRAY_PARTITION variable=ob type=complete
    int st = 0;                 // the step of the pixel
    px: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=7836192
        if (st == 0) {
            const vec_t<uact_t, IN> v = in.read();
            for (int i = 0; i < IN; i++) {
                #pragma HLS UNROLL
                x[i] = v.v[i];
            }
        }
        for (int k = 0; k < OUT - PO; k++) {
            #pragma HLS UNROLL
            ob[k] = ob[k + PO];
        }
        for (int j = 0; j < PO; j++) {
            #pragma HLS UNROLL
            acc_t sum = bl[st][j];
            for (int i = 0; i < IN; i++) {
                #pragma HLS UNROLL
                sum += mul<DSP, acc_t>(x[i], wl[st][j][i]);
            }
            ob[OUT - PO + j] = pointwise1d_out<RELU, SHIFT, ACT_MAX>::of(sum);
        }
        if (st == STEPS - 1) {
            vec_t<out_t, OUT> o;
            for (int i = 0; i < OUT; i++) {
                #pragma HLS UNROLL
                o.v[i] = ob[i];
            }
            out.write(o);
            st = 0;
        } else {
            st++;
        }
    }
}
