#pragma once

#include "conv3x3_stream.hpp"
#include "patch_grid.hpp"

// conv3x3_grouped: a 3x3 conv layer, PI input channels x PO output channels x
// KR rows of the 3x3 window a cycle: 3 x KR x PI x PO multipliers,
// IN/PI x OUT/PO x 3/KR cycles a pixel. With KR = 3 (the whole window a step)
// it is conv3x3_serial (read that first) with PO outputs a step instead of
// one, so a layer can be sized for any time between conv3x3_narrow's (all
// inputs at once) and conv3x3_serial's (one output at a time); KR = 1 (a row
// of the window a step) sizes it in steps of 3 multipliers instead of 9, for
// a chip with few DSPs. The same input and output as both: all channels of a
// pixel at once, raster order; out the sums, unscaled, or (RELU_SHIFT > 0)
// ReLU'd and rounded off RELU_SHIFT bits to int16 activations (sums_out,
// fixed_point.hpp), so no ReLU module is needed after it.
//
// The steps of a pixel: output group og outer, input group ig, then window
// rows kr inner; each adds PI x KR x 3 products into each of PO running sums.
// The weights: one wide memory word a step, its PO x PI x KR x 3 weights, at
// (og * IN/PI + ig) * 3/KR + kr. One memory a layer, not one per (PO, PI,
// tap): a thousand small memories made sp_unet_large_opt_direct too congested
// to route.
template<int PI, int PO, int KR = 3>
struct grouped_word_t {
    wgt_t v[PO][PI][KR][3];
};

template<int H, int W, int IN_CH, int OUT_CH, int PI, int PO, int KR, int RELU_SHIFT, bool DSP>
void conv3x3_grouped_patch(hls::stream<vec_t<act_t, IN_CH> >& in,
                           const grouped_word_t<PI, PO, KR> wl[(OUT_CH / PO) * (IN_CH / PI) * (3 / KR)],
                           const acc_t bl[OUT_CH / PO][PO],
                           hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT_CH> >& out) {
    static_assert(IN_CH % PI == 0 && OUT_CH % PO == 0, "PI must divide IN_CH, PO OUT_CH");
    static_assert(KR == 1 || KR == 3, "a step is one row of the window, or all three");
    constexpr int IG = IN_CH / PI, OG = OUT_CH / PO, NR = 3 / KR, STEPS = OG * IG * NR;
    constexpr int VECTORS = H * W, LAG = W + 1;
    typedef typename sums_out<RELU_SHIFT>::type out_t;

    act_t lb[2][W][IN_CH];      // the two previous input lines
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=3
    act_t win[3][3][IN_CH];     // the 3x3 window, in registers
    #pragma HLS ARRAY_PARTITION variable=win type=complete dim=0
    out_t pix[OUT_CH];          // the finished outputs of one pixel
    #pragma HLS ARRAY_PARTITION variable=pix type=complete
    acc_t run[PO];              // the sums of this output group so far
    #pragma HLS ARRAY_PARTITION variable=run type=complete

    int s = 0;                  // column of the next input vector
    int y = 0, x = 0;           // position of the window's centre
    int og = 0, ig = 0, kr = 0; // this step's output group, input group, window rows
    int g = 0;                  // input steps so far, the last W + 1 without input

    conv: for (int t = 0; t < (VECTORS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=lb type=inter false
        // Spelled out so that with one window step they are constants.
        const bool first_row = NR == 1 || kr == 0;
        const bool last_row = NR == 1 || kr == NR - 1;
        const bool first_step = og == 0 && ig == 0 && first_row;
        const bool last_in = ig == IG - 1 && last_row;
        const bool last_step = og == OG - 1 && last_in;

        if (first_step) {
            // A new window column: input rows r-2, r-1, r at column s.
            const bool have = g < VECTORS;
            vec_t<act_t, IN_CH> v;
            if (have) v = in.read();
            for (int p = 0; p < IN_CH; p++) {
                #pragma HLS UNROLL
                const act_t up2 = lb[0][s][p], up1 = lb[1][s][p];
                const act_t now = have ? v.v[p] : (act_t)0;
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
        }

        if (g >= LAG) {
            // Taps outside the image hold other rows' values: they count as zero.
            const bool row_ok[3] = {y > 0, true, y < H - 1};
            const bool col_ok[3] = {x > 0, true, x < W - 1};
            const int row0 = NR == 1 ? 0 : kr;   // the window row of this step's first row
            const grouped_word_t<PI, PO, KR> wd = wl[(og * IG + ig) * NR + (NR == 1 ? 0 : kr)];
            for (int j = 0; j < PO; j++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                for (int p = 0; p < PI; p++) {
                    #pragma HLS UNROLL
                    for (int r = 0; r < KR; r++) {
                        #pragma HLS UNROLL
                        const int ki = row0 + r;
                        for (int kj = 0; kj < 3; kj++) {
                            #pragma HLS UNROLL
                            const act_t a = (row_ok[ki] && col_ok[kj]) ? win[ki][kj][ig * PI + p] : (act_t)0;
                            sum += mul<DSP, acc_t>(a, wd.v[j][p][r][kj]);
                        }
                    }
                }
                const acc_t total = (ig == 0 && first_row ? bl[og][j] : run[j]) + sum;
                run[j] = total;
                // Rounded as its output group finishes: PO of these, not OUT_CH.
                const out_t done = sums_out<RELU_SHIFT>::of(total);
                for (int k = 0; k < OG; k++) {
                    #pragma HLS UNROLL
                    if (last_in && og == k) pix[k * PO + j] = done;
                }
            }
            if (last_step) {
                vec_t<out_t, OUT_CH> o;
                for (int c = 0; c < OUT_CH; c++) {
                    #pragma HLS UNROLL
                    o.v[c] = pix[c];
                }
                out.write(o);
            }
        }

        if (!last_row) {
            kr++;
        } else {
            kr = 0;
            if (ig < IG - 1) {
                ig++;
            } else {
                ig = 0;
                if (!last_step) {
                    og++;
                } else {
                    og = 0;
                    if (g >= LAG) {
                        if (x < W - 1) x++;
                        else if (y < H - 1) { x = 0; y++; }
                        else { x = 0; y = 0; }
                    }
                    g++;
                }
            }
        }
    }
}

// The layer's weights, a word a step: step (og * IN/PI + ig) * 3/KR + kr holds
// output channels og * PO .. + PO by input channels ig * PI .. + PI, window
// rows kr * KR .. + KR. A word's weights come from the ROM at once, in as many
// cycles as its ports take (HLS picks the II), once, when the layer starts.
// The biases [og][j].
template<int IN, int OUT, int PI, int PO, int KR, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT]>
void load_grouped_weights(grouped_word_t<PI, PO, KR> wl[(OUT / PO) * (IN / PI) * (3 / KR)],
                          acc_t bl[OUT / PO][PO]) {
    constexpr int IG = IN / PI, NR = 3 / KR, STEPS = (OUT / PO) * IG * NR;
    load_w: for (int st = 0; st < STEPS; st++) {
        #pragma HLS PIPELINE
        const int og = st / (IG * NR), ig = (st / NR) % IG, kr = st % NR;
        grouped_word_t<PI, PO, KR> word;
        for (int j = 0; j < PO; j++) {
            #pragma HLS UNROLL
            for (int p = 0; p < PI; p++) {
                #pragma HLS UNROLL
                for (int r = 0; r < KR; r++) {
                    #pragma HLS UNROLL
                    for (int kj = 0; kj < 3; kj++) {
                        #pragma HLS UNROLL
                        word.v[j][p][r][kj] = WEIGHT[og * PO + j][ig * PI + p][kr * KR + r][kj];
                    }
                }
            }
        }
        wl[st] = word;
    }
    load_b: for (int oc = 0; oc < OUT; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc / PO][oc % PO] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
    }
}

// The layer: the image is cut into PH x PW patches (patch_grid.hpp); the layer
// is H x W of each. WEIGHT and BIAS: the layer's quantized int8 arrays,
// template arguments so that they stay a constant ROM inside this process.
// BIAS_SHIFT: the sums' fraction bits minus the bias's. DSP: the 3 x KR x PI x
// PO multipliers in DSPs, or in LUTs. KR and RELU_SHIFT as above.
// The weight memory is block RAM when it is deep, LUTs (distributed RAM) when
// it is shallow, where block RAM would hold a few rows of a mostly empty
// width. HLS's impl takes only a literal word: one branch each, HLS keeps the
// one the depth picks.
template<int PH, int PW, int H, int W, int IN, int OUT, int PI, int PO, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT], bool DSP,
         int KR = 3, int RELU_SHIFT = 0>
void conv3x3_grouped(hls::stream<vec_t<act_t, IN> >& in,
                     hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT> >& out,
                     int height, int width) {
    constexpr int STEPS = (OUT / PO) * (IN / PI) * (3 / KR);
    acc_t bl[OUT / PO][PO];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete dim=2
    const int n = patch_count<PH, PW>(height, width);
    if (STEPS > 64) {
        grouped_word_t<PI, PO, KR> wl[STEPS];
        #pragma HLS AGGREGATE variable=wl compact=bit
        #pragma HLS BIND_STORAGE variable=wl type=ram_2p impl=bram
        load_grouped_weights<IN, OUT, PI, PO, KR, BIAS_SHIFT, WEIGHT, BIAS>(wl, bl);
        patches_bram: for (int k = 0; k < n; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=665
            conv3x3_grouped_patch<H, W, IN, OUT, PI, PO, KR, RELU_SHIFT, DSP>(in, wl, bl, out);
        }
    } else {
        grouped_word_t<PI, PO, KR> wl[STEPS];
        #pragma HLS AGGREGATE variable=wl compact=bit
        #pragma HLS BIND_STORAGE variable=wl type=ram_2p impl=lutram
        load_grouped_weights<IN, OUT, PI, PO, KR, BIAS_SHIFT, WEIGHT, BIAS>(wl, bl);
        patches: for (int k = 0; k < n; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=665
            conv3x3_grouped_patch<H, W, IN, OUT, PI, PO, KR, RELU_SHIFT, DSP>(in, wl, bl, out);
        }
    }
}
