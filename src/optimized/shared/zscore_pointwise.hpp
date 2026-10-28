#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "pixel_io.hpp"        // zscore12, pixel_lane
#include "conv1d_engine.hpp"   // uact_t, requant_relu8

// zscore_pointwise: a per-pixel network's first layer, a 1x1 conv on the
// z-scored bands, done on the raw words as they come from the reader: no
// spectrum is ever assembled. A pixel is WORDS words of WB raw bands; words
// FIRST .. WORDS - 1 are kept, so kept band k is word FIRST + k / WB, lane
// k % WB. Each cycle one word: its WB bands z-scored, as zscore_pixels does
// ((dn * ZA[k] + ZB[k]) >> FRAC, saturated at +-IN_MAX), and multiplied into
// all OUT output channels' sums at once. At the pixel's last word the sums are
// done: ReLU, rounded off SHIFT bits, saturated at ACT_MAX, out as one vector.
//
//   one word a cycle:  WB z-scores  ─►  WB x OUT multiply-adds  ─►  OUT sums
//
// WORDS cycles a pixel (the dropped words take theirs too: the reader sends
// them), WB z-score multipliers (ZDSP: in DSPs, or LUTs) and WB x OUT
// multipliers of 12 x 8 bits (DSP: in DSPs, or LUTs). The 12-bit z-scores are
// too wide to share a DSP between two products (conv1d_engine.hpp).
//
// WEIGHT and BIAS: the layer's int8 arrays, Conv2d's [OUT][IN][1][1], template
// arguments so that they stay a ROM inside this process; kept bands past IN
// (bands 110 and 111, the z-score's zeros) get weight 0. Each (band lane,
// output channel) has its own small memory, addressed by the word: the run-time
// choice is the address, never the bank. BIAS_SHIFT: the sums' fraction bits
// minus the bias's. LANE: which pixels (pixel_lane).
template<int WORDS, int FIRST, int WB, int FRAC, int IN_MAX,
         const int32_t (&ZA)[(WORDS - FIRST) * WB], const int64_t (&ZB)[(WORDS - FIRST) * WB],
         int IN, int OUT, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OUT][IN][1][1], const wgt_t (&BIAS)[OUT], bool ZDSP, bool DSP,
         typename LANE, typename WORD_T>
void zscore_pointwise(hls::stream<WORD_T>& raw, hls::stream<vec_t<uact_t, OUT> >& out, int height, int width) {
    constexpr int KEPT = WORDS - FIRST;           // kept words per pixel
    constexpr int BANDS = KEPT * WB;
    static_assert(IN <= BANDS, "the layer reads only kept bands");
    static_assert(BIAS_SHIFT >= 0, "the bias must not have more fraction bits than the sums");
    // The z-score constants and the weights, one small ROM per lane (and output
    // channel), addressed by the kept word.
    int32_t za[KEPT][WB];
    int64_t zb[KEPT][WB];
    wgt_t wl[KEPT][WB][OUT];
    acc_t bl[OUT];
    #pragma HLS ARRAY_PARTITION variable=za type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=zb type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=bl type=complete
    load_z: for (int k = 0; k < BANDS; k++) {
        za[k / WB][k % WB] = ZA[k];
        zb[k / WB][k % WB] = ZB[k];
        for (int oc = 0; oc < OUT; oc++) {
            #pragma HLS PIPELINE II=1
            const int src = k < IN ? k : IN - 1;  // never read past WEIGHT
            wl[k / WB][k % WB][oc] = k < IN ? WEIGHT[oc][src][0][0] : (wgt_t)0;
        }
    }
    load_b: for (int oc = 0; oc < OUT; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
    }

    const uint32_t pixels = LANE::count(height, width);
    acc_t acc[OUT];                               // the pixel's sums so far
    #pragma HLS ARRAY_PARTITION variable=acc type=complete
    int w = 0;                                    // word of the pixel
    word: for (uint32_t t = 0; t < pixels * WORDS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=15 max=9795240
        const WORD_T word = raw.read();
        vec_t<uact_t, OUT> o;
        if (w >= FIRST) {
            const int kw = w - FIRST;
            int16_t z[WB];
            #pragma HLS ARRAY_PARTITION variable=z type=complete
            for (int p = 0; p < WB; p++) {
                #pragma HLS UNROLL
                z[p] = zscore12<FRAC, IN_MAX, ZDSP>(word[p], za[kw][p], zb[kw][p]);
            }
            for (int oc = 0; oc < OUT; oc++) {
                #pragma HLS UNROLL
                acc_t part = 0;
                for (int p = 0; p < WB; p++) {
                    #pragma HLS UNROLL
                    part += mul<DSP, acc_t>(z[p], wl[kw][p][oc]);
                }
                const acc_t total = (w == FIRST ? bl[oc] : acc[oc]) + part;
                acc[oc] = total;
                o.v[oc] = requant_relu8<SHIFT, ACT_MAX>(total);
            }
        }
        if (w == WORDS - 1) {
            out.write(o);
            w = 0;
        } else {
            w++;
        }
    }
}
