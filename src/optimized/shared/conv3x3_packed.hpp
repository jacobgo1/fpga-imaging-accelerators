#pragma once

#include "conv3x3_stream.hpp"
#include "patch_grid.hpp"

// conv3x3_stream for int8 activations, two multiplies per DSP (AMD white paper
// WP486, "Deep Learning with INT8 Optimization on Xilinx Devices").
//
// Two output channels oc, oc + 1 take the same activation a, so their weights
// are packed into one number,
//     (w[oc + 1] * 2^SHIFT + w[oc]) * a = w[oc + 1] * a * 2^SHIFT + w[oc] * a,
// and one multiply gives both products: the low one in the bottom SHIFT bits,
// the high one above. A sum of such products is the two channels' sums the
// same way, as long as the low sum stays inside SHIFT signed bits: |a| <= 127
// and |w| <= 128 make a product at most 16256, so GROUP of them must stay below
// 2^(SHIFT - 1). Each group is split into its two sums, then those are added
// as usual. The result is exact: bit for bit what two separate multiplies give.
//
// The packed weight takes SHIFT + 9 bits, so the DSP's wide input sets SHIFT
// (the boards' conv3x3_wide pick it):
//   - DSP48E2 (UltraScale+, the ZCU104), 27 x 18 bits: SHIFT 18, groups of 8;
//   - DSP48E1 (7-series, the Zynq-7000s), 25 x 18 bits: SHIFT 16, groups of 2.
// a's -128 must never occur: the producer saturates at +-127.
//
// The packed weights are computed once, when the layer starts
// (load_packed_weights), not next to the multiply. There HLS (2025.2) put the
// packing addition into the DSP's pre-adder, sized for the two weights' own
// widths, and the hardware's sums came out slightly wrong in co-simulation
// while the C++ was right. A finished packed number times a is a plain
// multiply-add, which it gets right.
//
// So it has no DSP switch: in LUTs, a packed multiply and its split cost more
// than two plain ones (conv3x3_stream with DSP = false). Each multiply is
// bound to a DSP.
//
// Everything else is conv3x3_stream with STEPS = 1 (read that first): P
// channels a cycle, chunk by chunk, OUT_CH (even) outputs per pixel in the
// last chunk. OUT_CH / 2 x P x 9 DSPs.

#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
template<int SHIFT> using pack_w_t = ap_int<SHIFT + 9>;   // two weights in one DSP input
typedef ap_int<48> pack_s_t;                              // a group's sum, the DSP's accumulator
#else
template<int SHIFT> using pack_w_t = int64_t;   // plain g++: wide enough for the same values,
typedef int64_t pack_s_t;                       // products included
#endif

// A group's sum -> the low channel's part: its bottom SHIFT bits, sign-extended.
template<int SHIFT>
acc_t pack_low(int64_t s) {
    #pragma HLS INLINE
    return (acc_t)((uint32_t)(uint64_t)s << (32 - SHIFT)) >> (32 - SHIFT);
}

// ... and the high channel's: the rest, plus the 1 the low part borrowed when
// it is negative (its sign is bit SHIFT - 1).
template<int SHIFT>
acc_t pack_high(int64_t s) {
    #pragma HLS INLINE
    return (acc_t)(s >> SHIFT) + (acc_t)((s >> (SHIFT - 1)) & 1);
}

// load_conv_weights for conv3x3_packed: output channels 2j and 2j + 1 packed
// into wp[j], w[2j + 1] * 2^SHIFT + w[2j]; IN_PAD > IN_CH pads with zeros.
template<int OUT_CH, int IN_CH, int IN_PAD, int SHIFT, int BIAS_SHIFT>
void load_packed_weights(const wgt_t w[OUT_CH][IN_CH][3][3], const wgt_t b[OUT_CH],
                         pack_w_t<SHIFT> wp[OUT_CH / 2][IN_PAD][3][3], acc_t bl[OUT_CH]) {
    static_assert(IN_PAD >= IN_CH && OUT_CH % 2 == 0, "IN_PAD >= IN_CH, output channels in pairs");
    load_w: for (int j = 0; j < OUT_CH / 2; j++)
        for (int ic = 0; ic < IN_PAD; ic++)
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++) {
                    #pragma HLS PIPELINE II=1
                    const int src = ic < IN_CH ? ic : IN_CH - 1;   // never read past w
                    const wgt_t hi = ic < IN_CH ? w[2 * j + 1][src][ki][kj] : (wgt_t)0;
                    const wgt_t lo = ic < IN_CH ? w[2 * j][src][ki][kj] : (wgt_t)0;
                    wp[j][ic][ki][kj] = (pack_w_t<SHIFT>)hi * (pack_w_t<SHIFT>)(1 << SHIFT) + (pack_w_t<SHIFT>)lo;
                }
    load_b: for (int oc = 0; oc < OUT_CH; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc] = (acc_t)b[oc] * ((acc_t)1 << BIAS_SHIFT);
    }
}

// The engine, on one patch. wp partitioned as conv3x3_stream's wl with
// STEPS = 1: complete in dims 1, 3, 4, cyclic by P in dim 2.
template<int H, int W, int IN_CH, int OUT_CH, int P, int SHIFT, int GROUP>
void conv3x3_packed(hls::stream<vec_t<act8_t, P> >& in,
                    const pack_w_t<SHIFT> wp[OUT_CH / 2][IN_CH][3][3], const acc_t bl[OUT_CH],
                    hls::stream<vec_t<acc_t, OUT_CH> >& out) {
    static_assert(IN_CH % P == 0, "IN_CH must be a multiple of P");
    static_assert(P % GROUP == 0, "P must be a multiple of GROUP");
    static_assert(OUT_CH % 2 == 0, "output channels go in pairs");
    static_assert(GROUP * 127 * 128 < (1 << (SHIFT - 1)), "a group's low sum must stay inside SHIFT signed bits");
    constexpr int CHUNKS = IN_CH / P;
    constexpr int VECTORS = CHUNKS * H * W, LAG = W + 1;

    acc_t acc[CHUNKS > 1 ? H : 1][CHUNKS > 1 ? W : 1][OUT_CH];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete dim=3
    act8_t lb[2][W][P];      // the two previous input lines
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=3
    act8_t win[3][3][P];     // the 3x3 window, in registers
    #pragma HLS ARRAY_PARTITION variable=win type=complete dim=0

    int s = 0;                  // column of the next input vector
    int co = 0, y = 0, x = 0;   // chunk and position of the window's centre
    int g = 0;                  // input steps so far, the last W + 1 without input

    conv: for (int t = 0; t < VECTORS + LAG; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=acc type=inter false
        #pragma HLS DEPENDENCE variable=lb type=inter false
        const bool first_chunk = CHUNKS == 1 || co == 0;
        const bool last_chunk = CHUNKS == 1 || co == CHUNKS - 1;
        const int chunk = CHUNKS == 1 ? 0 : co;

        // A new window column: input rows r-2, r-1, r at column s.
        const bool have = g < VECTORS;
        vec_t<act8_t, P> v;
        if (have) v = in.read();
        for (int p = 0; p < P; p++) {
            #pragma HLS UNROLL
            const act8_t up2 = lb[0][s][p], up1 = lb[1][s][p];
            const act8_t now = have ? v.v[p] : (act8_t)0;
            lb[0][s][p] = up1;
            lb[1][s][p] = now;
            for (int ki = 0; ki < 3; ki++) {
                #pragma HLS UNROLL
                win[ki][0][p] = win[ki][1][p];
                win[ki][1][p] = win[ki][2][p];
            }
            win[0][2][p] = up2;
            win[1][2][p] = up1;
            win[2][2][p] = now;
        }
        s = (s == W - 1) ? 0 : s + 1;

        if (g >= LAG) {
            const bool row_ok[3] = {y > 0, true, y < H - 1};
            const bool col_ok[3] = {x > 0, true, x < W - 1};
            vec_t<acc_t, OUT_CH> o;
            for (int j = 0; j < OUT_CH / 2; j++) {
                #pragma HLS UNROLL
                acc_t lo = 0, hi = 0;   // channels 2j and 2j + 1
                for (int p0 = 0; p0 < P; p0 += GROUP) {
                    #pragma HLS UNROLL
                    for (int ki = 0; ki < 3; ki++) {
                        #pragma HLS UNROLL
                        for (int kj = 0; kj < 3; kj++) {
                            #pragma HLS UNROLL
                            pack_s_t sum = 0;
                            for (int p = p0; p < p0 + GROUP; p++) {
                                #pragma HLS UNROLL
                                const act8_t a = (row_ok[ki] && col_ok[kj]) ? win[ki][kj][p] : (act8_t)0;
                                const pack_s_t prod = wp[j][chunk * P + p][ki][kj] * a;
                                #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                                sum += prod;
                            }
                            lo += pack_low<SHIFT>((int64_t)sum);
                            hi += pack_high<SHIFT>((int64_t)sum);
                        }
                    }
                }
                const acc_t lo_total = (first_chunk ? bl[2 * j] : acc[y][x][2 * j]) + lo;
                const acc_t hi_total = (first_chunk ? bl[2 * j + 1] : acc[y][x][2 * j + 1]) + hi;
                if (last_chunk) {
                    o.v[2 * j] = lo_total;
                    o.v[2 * j + 1] = hi_total;
                } else {
                    acc[y][x][2 * j] = lo_total;
                    acc[y][x][2 * j + 1] = hi_total;
                }
            }
            if (last_chunk) out.write(o);
            if (x < W - 1) x++;
            else if (y < H - 1) { x = 0; y++; }
            else { x = 0; y = 0; co++; }
        }
        g++;
    }
}

// The layer, for a board's conv3x3_wide: P input channels x every output
// channel per cycle, chunk by chunk. OUT / 2 x P x 9 DSPs, IN_PAD / P x H x W
// + W + 1 cycles per patch. The input channels are padded with zero weights up
// to IN_PAD, a multiple of P: the producer sends IN_PAD.
//
// The image is cut into PH x PW patches (patch_grid.hpp); the layer is H x W
// of each. WEIGHT and BIAS: the layer's quantized int8 arrays, template
// arguments so that they stay a constant ROM inside this process. BIAS_SHIFT:
// the sums' fraction bits minus the bias's. Out: every pixel's OUT sums,
// unscaled, raster order.
template<int PH, int PW, int H, int W, int IN, int OUT, int P, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT], int SHIFT, int GROUP>
void conv3x3_packed_layer(hls::stream<vec_t<act8_t, P> >& in, hls::stream<vec_t<acc_t, OUT> >& out,
                          int height, int width) {
    constexpr int IN_PAD = (IN + P - 1) / P * P;
    pack_w_t<SHIFT> wp[OUT / 2][IN_PAD][3][3];  // channel pairs, packed
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wp type=cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=4
    acc_t bl[OUT];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete

    load_packed_weights<OUT, IN, IN_PAD, SHIFT, BIAS_SHIFT>(WEIGHT, BIAS, wp, bl);
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_packed<H, W, IN_PAD, OUT, P, SHIFT, GROUP>(in, wp, bl, out);
    }
}
