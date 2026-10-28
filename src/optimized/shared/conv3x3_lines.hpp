#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// conv3x3_lines: a 3x3 conv layer for a chip short of flip-flops and LUTs: the
// 3x3 window lives in memory, not registers. The same input and output as
// conv3x3_grouped (all channels of a pixel at once, raster order; out the sums,
// or, RELU_SHIFT > 0, ReLU'd int16 activations rounded as each output group
// finishes), the same answers; what differs is where the window is.
//
// conv3x3_grouped keeps 9 x IN activations in registers and picks each step's
// channels out of them: on the Zynq-7020 the window registers of SpecPW-UNet
// medium's ten convs alone would be 45 k of its 106 k flip-flops. Here the last
// three input rows are three memories, a row each, words of PI channels:
// row r in memory r mod 3, at column * IN/PI + input group. A step reads the
// three at one address, one column of the window for PI channels, and adds KT of
// its taps (KT = 3: the whole column; KT = 1: one tap, the row picked from the
// three reads), times PO output channels: KT x PI x PO multipliers,
// 9/KT x IN/PI x OUT/PO steps a pixel. Steps: output group, input group, then
// the window's taps, inner (its columns, then for one tap a step its rows).
//
// The input pixel comes in at a pixel's first step and is written to its row's
// memory group by group, one a step in its first IN/PI steps (a run-time choice
// of which memory's write runs: each memory has one write and one read at fixed
// places). The window's centre trails the input by a row and two pixels, so
// every value it reads was written at least a pixel earlier: a memory's write
// and read never meet, whatever the pipeline's depth. (H x W + W + 2) x steps
// cycles a patch. Taps outside the image hold other rows' values: the centre's
// position zeroes them.
template<int PI, int PO, int KT>
struct lines_word_t {
    wgt_t v[PO][PI][KT];
};

template<int H, int W, int IN_CH, int OUT_CH, int PI, int PO, int KT, int RELU_SHIFT, bool DSP>
void conv3x3_lines_patch(hls::stream<vec_t<act_t, IN_CH> >& in,
                         const lines_word_t<PI, PO, KT> wl[(OUT_CH / PO) * (IN_CH / PI) * (9 / KT)],
                         const acc_t bl[OUT_CH / PO][PO],
                         hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT_CH> >& out) {
    static_assert(IN_CH % PI == 0 && OUT_CH % PO == 0, "PI must divide IN_CH, PO OUT_CH");
    static_assert(KT == 1 || KT == 3, "a step is one tap, or a column of the window");
    constexpr int IG = IN_CH / PI, OG = OUT_CH / PO, NT = 9 / KT, STEPS = OG * IG * NT;
    constexpr int VECTORS = H * W, LAG = W + 2;
    typedef typename sums_out<RELU_SHIFT>::type out_t;
    struct word_t { act_t v[PI]; };

    word_t rows[3][W * IG];     // the last three input rows, a memory each
    #pragma HLS ARRAY_PARTITION variable=rows type=complete dim=1
    #pragma HLS AGGREGATE variable=rows compact=bit
    act_t x_in[IN_CH];          // this pixel's input, written to its row a group a step
    #pragma HLS ARRAY_PARTITION variable=x_in type=complete
    out_t pix[OUT_CH];          // the finished outputs of one pixel
    #pragma HLS ARRAY_PARTITION variable=pix type=complete
    acc_t run[PO];              // the sums of this output group so far
    #pragma HLS ARRAY_PARTITION variable=run type=complete

    int s = 0, wr = 0;          // the input's column, and its row's memory (row mod 3)
    int y = 0, x = 0;           // position of the window's centre
    int top = 2;                // the memory of the centre's row above: (y - 1) mod 3
    // This step's output group, input group, and the window's column kc and (one
    // tap a step) row kr: counters, not nt / 3 and nt % 3, which HLS built as a
    // 36-cycle divider in the loop. wa: the step's weight word, one after another.
    int og = 0, ig = 0, kc = 0, kr = 0, wa = 0;
    int g = 0;                  // input pixels so far, the last W + 2 without input

    conv: for (int t = 0; t < (VECTORS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        // A row's word is written a pixel or more before the centre reads it.
        #pragma HLS DEPENDENCE variable=rows type=inter false
        const bool first_tap = kc == 0 && (KT == 3 || kr == 0);
        const bool last_tap = kc == 2 && (KT == 3 || kr == 2);
        const bool first_step = og == 0 && ig == 0 && first_tap;
        const bool last_in = ig == IG - 1 && last_tap;
        const bool last_step = og == OG - 1 && last_in;
        const bool have = g < VECTORS;

        if (first_step && have) {
            const vec_t<act_t, IN_CH> v = in.read();
            for (int c = 0; c < IN_CH; c++) {
                #pragma HLS UNROLL
                x_in[c] = v.v[c];
            }
        }
        if (og == 0 && first_tap && have) {
            // Input group ig of this pixel into its row's memory.
            word_t wd;
            for (int p = 0; p < PI; p++) {
                #pragma HLS UNROLL
                wd.v[p] = x_in[(IG == 1 ? 0 : ig * PI) + p];
            }
            for (int m = 0; m < 3; m++) {
                #pragma HLS UNROLL
                if (wr == m) rows[m][s * IG + ig] = wd;
            }
        }

        if (g >= LAG) {
            const int col = x + kc - 1;
            const int addr = (col < 0 ? 0 : col > W - 1 ? W - 1 : col) * IG + ig;
            word_t rd[3];
            #pragma HLS ARRAY_PARTITION variable=rd type=complete
            for (int m = 0; m < 3; m++) {
                #pragma HLS UNROLL
                rd[m] = rows[m][addr];
            }
            const bool row_ok[3] = {y > 0, true, y < H - 1};
            const bool col_ok = (kc != 0 || x > 0) && (kc != 2 || x < W - 1);
            // The window's rows y - 1, y, y + 1 are in memories top, top + 1, top + 2 (mod 3).
            act_t a[KT][PI];
            #pragma HLS ARRAY_PARTITION variable=a type=complete dim=0
            for (int r = 0; r < KT; r++) {
                #pragma HLS UNROLL
                const int ki = KT == 3 ? r : kr;
                const int m = top + ki >= 3 ? top + ki - 3 : top + ki;
                for (int p = 0; p < PI; p++) {
                    #pragma HLS UNROLL
                    const act_t v = m == 0 ? rd[0].v[p] : m == 1 ? rd[1].v[p] : rd[2].v[p];
                    a[r][p] = (row_ok[ki] && col_ok) ? v : (act_t)0;
                }
            }
            const lines_word_t<PI, PO, KT> wd = wl[wa];
            for (int j = 0; j < PO; j++) {
                #pragma HLS UNROLL
                acc_t sum = 0;
                for (int r = 0; r < KT; r++) {
                    #pragma HLS UNROLL
                    for (int p = 0; p < PI; p++) {
                        #pragma HLS UNROLL
                        sum += mul<DSP, acc_t>(a[r][p], wd.v[j][p][r]);
                    }
                }
                const acc_t total = (ig == 0 && first_tap ? bl[og][j] : run[j]) + sum;
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

        // The next step: the window's taps (columns, then for one tap a step the
        // rows), then the input group, then the output group, then the pixel.
        wa = last_step ? 0 : wa + 1;
        if (kc < 2) {
            kc++;
        } else if (KT == 1 && kr < 2) {
            kc = 0;
            kr++;
        } else {
            kc = 0;
            kr = 0;
            if (ig < IG - 1) {
                ig++;
            } else {
                ig = 0;
                if (!last_step) {
                    og++;
                } else {
                    og = 0;
                    // The next pixel: the input's column and row, the centre's.
                    if (s < W - 1) s++;
                    else { s = 0; wr = wr == 2 ? 0 : wr + 1; }
                    if (g >= LAG) {
                        if (x < W - 1) x++;
                        else { x = 0; top = top == 2 ? 0 : top + 1; y = y < H - 1 ? y + 1 : 0; }
                    }
                    g++;
                }
            }
        }
    }
}

// The layer's weights, a word a step: step (og * IN/PI + ig) * 9/KT + nt holds
// output channels og * PO .. + PO by input channels ig * PI .. + PI, the
// window's taps nt (KT = 1: tap nt, row nt / 3, column nt % 3) or column nt
// (KT = 3, rows 0-2), in the order the steps take them. A word's weights come from the ROM at once, in as many
// cycles as its ports take, once, when the layer starts. The biases [og][j].
template<int IN, int OUT, int PI, int PO, int KT, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT]>
void load_lines_weights(lines_word_t<PI, PO, KT> wl[(OUT / PO) * (IN / PI) * (9 / KT)], acc_t bl[OUT / PO][PO]) {
    constexpr int IG = IN / PI, NT = 9 / KT;
    // Nested loops, not st / ... and st % ...: no dividers, even at load time.
    int st = 0;
    load_w: for (int og = 0; og < OUT / PO; og++)
        for (int ig = 0; ig < IG; ig++)
            for (int nt = 0; nt < NT; nt++) {
                #pragma HLS PIPELINE
                lines_word_t<PI, PO, KT> word;
                for (int j = 0; j < PO; j++) {
                    #pragma HLS UNROLL
                    for (int p = 0; p < PI; p++) {
                        #pragma HLS UNROLL
                        for (int r = 0; r < KT; r++) {
                            #pragma HLS UNROLL
                            // one tap a step: tap nt is row nt / 3, column nt % 3, without dividing
                            const int ki = KT == 3 ? r : nt >= 6 ? 2 : nt >= 3 ? 1 : 0;
                            const int kj = KT == 3 ? nt : nt - 3 * ki;
                            word.v[j][p][r] = WEIGHT[og * PO + j][ig * PI + p][ki][kj];
                        }
                    }
                }
                wl[st] = word;
                st++;
            }
    load_b: for (int oc = 0; oc < OUT; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc / PO][oc % PO] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
    }
}

// The layer: the image is cut into PH x PW patches (patch_grid.hpp); the layer
// is H x W of each. WEIGHT and BIAS: the layer's quantized int8 arrays,
// template arguments so that they stay a constant ROM inside this process.
// BIAS_SHIFT: the sums' fraction bits minus the bias's. DSP: the KT x PI x PO
// multipliers in DSPs, or in LUTs. The weights: a deep, narrow memory, block RAM.
template<int PH, int PW, int H, int W, int IN, int OUT, int PI, int PO, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT], bool DSP, int KT, int RELU_SHIFT>
void conv3x3_lines(hls::stream<vec_t<act_t, IN> >& in,
                   hls::stream<vec_t<typename sums_out<RELU_SHIFT>::type, OUT> >& out,
                   int height, int width) {
    constexpr int STEPS = (OUT / PO) * (IN / PI) * (9 / KT);
    lines_word_t<PI, PO, KT> wl[STEPS];
    #pragma HLS AGGREGATE variable=wl compact=bit
    #pragma HLS BIND_STORAGE variable=wl type=ram_2p impl=bram
    acc_t bl[OUT / PO][PO];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete dim=2
    load_lines_weights<IN, OUT, PI, PO, KT, BIAS_SHIFT, WEIGHT, BIAS>(wl, bl);
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_lines_patch<H, W, IN, OUT, PI, PO, KT, RELU_SHIFT, DSP>(in, wl, bl, out);
    }
}
