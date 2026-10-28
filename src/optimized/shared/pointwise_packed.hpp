#pragma once

#include "conv3x3_packed.hpp"
#include "patch_grid.hpp"

// pointwise_packed: a 1x1 conv layer on int8 inputs (the z-scored bands, a
// spectral stem's first layer), two multiplies per DSP as in
// conv3x3_packed.hpp (read that first): output channels 2j and 2j + 1 share
// each band, so their weights are packed into one number and one multiply
// gives both products. ReLU, then rounded off SHIFT bits to int16 activations.
//
// In: a pixel's bands in chunks of P, one chunk a cycle (pixel_buffer), the
// last one padded with zero bands. Out: the pixel's OUT activations, once its
// last chunk is in. OUT / 2 x P DSPs, IN_PAD / P cycles a pixel. The pixel
// order does not matter (a 1x1 conv): raster or block order.
//
// The image is cut into PH x PW patches (patch_grid.hpp), PIX pixels each.
// WEIGHT and BIAS: the layer's int8 arrays, template arguments so that they
// stay a ROM inside this process. BIAS_SHIFT: the sums' fraction bits minus
// the bias's. PSHIFT and GROUP: the packing, the board's (zcu104/pointwise_wide.hpp).
//
// OPASS: the channel pairs in OPASS passes over each chunk (the chunk read once
// and kept), OUT / 2 / OPASS x P DSPs and IN_PAD / P x OPASS cycles a pixel,
// for a chip with few DSPs; 1: all pairs at once.
template<int PH, int PW, int PIX, int IN, int OUT, int P, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][1][1], const wgt_t (&BIAS)[OUT], int PSHIFT, int GROUP, int OPASS = 1>
void pointwise_packed(hls::stream<vec_t<act8_t, P> >& in, hls::stream<vec_t<act_t, OUT> >& out,
                      int height, int width) {
    static_assert(OUT % 2 == 0, "output channels go in pairs");
    static_assert((OUT / 2) % OPASS == 0, "OPASS must divide the channel pairs");
    static_assert(P % GROUP == 0, "P must be a multiple of GROUP");
    static_assert(GROUP * 127 * 128 < (1 << (PSHIFT - 1)), "a group's low sum must stay inside PSHIFT signed bits");
    constexpr int CHUNKS = (IN + P - 1) / P, IN_PAD = CHUNKS * P, PP = OUT / 2 / OPASS;

    // Pair o * PP + jj's packed weights at [jj][chunk * OPASS + o][lane], zero past IN.
    pack_w_t<PSHIFT> wp[PP][CHUNKS * OPASS][P];
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=3
    acc_t bl[OUT];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete
    load_w: for (int j = 0; j < OUT / 2; j++)
        for (int ic = 0; ic < IN_PAD; ic++) {
            #pragma HLS PIPELINE II=1
            const int src = ic < IN ? ic : IN - 1;   // never read past WEIGHT
            const wgt_t hi = ic < IN ? WEIGHT[2 * j + 1][src][0][0] : (wgt_t)0;
            const wgt_t lo = ic < IN ? WEIGHT[2 * j][src][0][0] : (wgt_t)0;
            wp[j % PP][(ic / P) * OPASS + j / PP][ic % P] =
                (pack_w_t<PSHIFT>)hi * (pack_w_t<PSHIFT>)(1 << PSHIFT) + (pack_w_t<PSHIFT>)lo;
        }
    load_b: for (int oc = 0; oc < OUT; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
    }

    acc_t acc[OUT];             // the pixel's sums so far
    #pragma HLS ARRAY_PARTITION variable=acc type=complete
    act_t ob[OUT];              // its activations, a pass's pairs at a time
    #pragma HLS ARRAY_PARTITION variable=ob type=complete
    act8_t x[P];                // this chunk, kept for its passes
    #pragma HLS ARRAY_PARTITION variable=x type=complete
    const int pixels = patch_count<PH, PW>(height, width) * PIX;
    int c = 0, o = 0;           // the chunk of the pixel, the pass over it
    constexpr int TRIPS = PIX * CHUNKS * OPASS, MAX_TRIPS = 665 * TRIPS;  // a patch; a 598 x 1092 capture
    (void)TRIPS; (void)MAX_TRIPS;   // only the pragma uses them
    px: for (int t = 0; t < pixels * CHUNKS * OPASS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=TRIPS max=MAX_TRIPS
        if (o == 0) {
            const vec_t<act8_t, P> v = in.read();
            for (int p = 0; p < P; p++) {
                #pragma HLS UNROLL
                x[p] = v.v[p];
            }
        }
        for (int jj = 0; jj < PP; jj++) {
            #pragma HLS UNROLL
            acc_t lo = 0, hi = 0;   // channels 2j and 2j + 1, j = o * PP + jj
            for (int p0 = 0; p0 < P; p0 += GROUP) {
                #pragma HLS UNROLL
                pack_s_t sum = 0;
                for (int p = p0; p < p0 + GROUP; p++) {
                    #pragma HLS UNROLL
                    const pack_s_t prod = wp[jj][c * OPASS + o][p] * x[p];
                    #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                    sum += prod;
                }
                lo += pack_low<PSHIFT>((int64_t)sum);
                hi += pack_high<PSHIFT>((int64_t)sum);
            }
            // The pass's pair j = o * PP + jj: its sums so far picked by o, one
            // adder and one rounding unit a lane whatever OPASS is; written back
            // by the pass whose turn it is (a run-time choice of which statement runs).
            const int j = o * PP + jj;
            const acc_t lo_total = (c == 0 ? bl[2 * j] : acc[2 * j]) + lo;
            const acc_t hi_total = (c == 0 ? bl[2 * j + 1] : acc[2 * j + 1]) + hi;
            const act_t lo_act = requant_relu<SHIFT>(lo_total), hi_act = requant_relu<SHIFT>(hi_total);
            for (int k = 0; k < OPASS; k++) {
                #pragma HLS UNROLL
                if (o == k) {
                    acc[2 * (k * PP + jj)] = lo_total;
                    acc[2 * (k * PP + jj) + 1] = hi_total;
                    if (c == CHUNKS - 1) {
                        ob[2 * (k * PP + jj)] = lo_act;
                        ob[2 * (k * PP + jj) + 1] = hi_act;
                    }
                }
            }
        }
        if (c == CHUNKS - 1 && o == OPASS - 1) {
            vec_t<act_t, OUT> out_v;
            for (int oc = 0; oc < OUT; oc++) {
                #pragma HLS UNROLL
                out_v.v[oc] = ob[oc];
            }
            out.write(out_v);
        }
        if (o < OPASS - 1) {
            o++;
        } else {
            o = 0;
            c = c < CHUNKS - 1 ? c + 1 : 0;
        }
    }
}
