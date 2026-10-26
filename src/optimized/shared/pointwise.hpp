#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// pointwise: a 1x1 conv layer on int16 activations, pixel by pixel: every
// input channel of a pixel at once, PO output channels a cycle, OUT/PO cycles
// a pixel, IN x PO multipliers (DSP: in DSPs, or LUTs). The weights: one small
// memory per (input channel, output lane), OUT/PO deep.
//
// RELU: each output ReLU'd and rounded off SHIFT bits to an int16 activation
// (out: act_t); otherwise the plain sums, bias included (out: acc_t), for the
// final layer's scores. G pixels go out together: G = 4 gathers each 2x2
// block's pixels into one vector, for a layer in block order
// (wino_conv3x3.hpp) when the pixels come in block order.
//
// The image is cut into PH x PW patches (patch_grid.hpp), PIX pixels each.
// WEIGHT and BIAS: the layer's int8 arrays, template arguments so that they
// stay a ROM inside this process. BIAS_SHIFT: the sums' fraction bits minus
// the bias's.
template<bool RELU, int SHIFT> struct pointwise_out {
    typedef act_t type;
    static type of(acc_t v) {
        #pragma HLS INLINE
        return requant_relu<SHIFT>(v);
    }
};
template<int SHIFT> struct pointwise_out<false, SHIFT> {
    typedef acc_t type;
    static type of(acc_t v) {
        #pragma HLS INLINE
        return v;
    }
};

template<int PH, int PW, int PIX, int IN, int OUT, int PO, int G, bool RELU, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][1][1], const wgt_t (&BIAS)[OUT], bool DSP>
void pointwise(hls::stream<vec_t<act_t, IN> >& in,
               hls::stream<vec_t<typename pointwise_out<RELU, SHIFT>::type, G * OUT> >& out,
               int height, int width) {
    static_assert(OUT % PO == 0 && PIX % G == 0, "PO must divide OUT, G the pixels");
    typedef typename pointwise_out<RELU, SHIFT>::type out_t;
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

    act_t x[IN];                // this pixel
    #pragma HLS ARRAY_PARTITION variable=x type=complete
    out_t ob[G * OUT];          // the G pixels' outputs
    #pragma HLS ARRAY_PARTITION variable=ob type=complete
    const int pixels = patch_count<PH, PW>(height, width) * PIX;
    int st = 0, g = 0;          // the step of the pixel, the pixel of the G
    constexpr int TRIPS = PIX * STEPS, MAX_TRIPS = 665 * TRIPS;   // a patch; a 598 x 1092 capture
    (void)TRIPS; (void)MAX_TRIPS;   // only the pragma uses them
    px: for (int t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=TRIPS max=MAX_TRIPS
        if (st == 0) {
            const vec_t<act_t, IN> v = in.read();
            for (int i = 0; i < IN; i++) {
                #pragma HLS UNROLL
                x[i] = v.v[i];
            }
        }
        for (int j = 0; j < PO; j++) {
            #pragma HLS UNROLL
            acc_t sum = bl[st][j];
            for (int i = 0; i < IN; i++) {
                #pragma HLS UNROLL
                sum += mul<DSP, acc_t>(x[i], wl[st][j][i]);
            }
            const out_t y = pointwise_out<RELU, SHIFT>::of(sum);
            for (int s = 0; s < STEPS; s++) {
                #pragma HLS UNROLL
                for (int q = 0; q < G; q++) {
                    #pragma HLS UNROLL
                    if (st == s && g == q) ob[q * OUT + s * PO + j] = y;
                }
            }
        }
        if (st == STEPS - 1 && g == G - 1) {
            vec_t<out_t, G * OUT> o;
            for (int i = 0; i < G * OUT; i++) {
                #pragma HLS UNROLL
                o.v[i] = ob[i];
            }
            out.write(o);
        }
        if (st < STEPS - 1) {
            st++;
        } else {
            st = 0;
            g = g < G - 1 ? g + 1 : 0;
        }
    }
}
