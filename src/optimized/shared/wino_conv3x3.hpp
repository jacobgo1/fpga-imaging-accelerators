#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// wino_conv3x3: a same-padded (zero) 3x3 conv layer and its ReLU, as Winograd
// F(2x2, 3x3) (Lavin and Gray, "Fast Algorithms for Convolutional Neural
// Networks", 2016): a 2x2 block of outputs from the 4x4 tile of inputs around
// it, with 16 multiplies per input x output channel instead of 36,
//
//     4 Y = A^T [ U ⊙ V ] A,   V = B^T d B (the tile),   U = (2G) g (2G)^T (the weights).
//
// In integers it is exact. U is 4 times the textbook transform, so every
// coefficient is an integer, and A^T [U ⊙ V] A is exactly 4 x the conv's sum:
// the >> 2 drops only zeros. So the answer is bit for bit the direct conv's.
//
// The sizes, all checked by the numbers: the inputs are ReLU'd activations,
// 0 .. 32767, so a V (a sum of four with signs) lies in -65534 .. 131068, 18
// signed bits; |U| <= 9 x 128 = 1152, 12 bits; their product fits one DSP48E2
// (18 x 27 bits). The sums over the input channels are kept modulo 2^34: the
// output's 4Y modulo 2^34 is then Y modulo 2^32, exactly what the direct
// engines' int32 sums hold.
//
// The streams are in 2x2-block order: an image of H x W pixels is a grid of
// H/2 x W/2 blocks, raster order, and a block is one vector of its 4 pixels x
// CH channels, v[(2 dy + dx) * CH + c]. Then maxpool is a block on its own,
// and Winograd's 2x2 outputs are a block: no reordering in between.
//
// wino_up_conv3x3 is the same engine for an UpBlock: nearest-upsample x2, then
// a 3x3 conv. The upsampled 4x4 tile around an output block has rows (and
// columns) a, b, b, c of the low-res map, so V's row 2 and column 2 are zero:
// 9 multiplies per block instead of 16 (and 36 direct), straight from the
// low-res pixels, which come in raster order, one vector per pixel.
//
// How it runs. Each block (or low-res pixel) is a period of STEPS = IN/PI x
// OUT/PO steps, one per cycle (II = 1): output group og outer, input group ig
// inner. A step multiplies PI input channels' V by PO output channels' U,
// NV x PI x PO multipliers (NV = 16, or 9 up), and adds into the PO
// accumulators M; the last ig turns them into the block's PO outputs.
//
// The window: four column slots (one memory each, IN/PI deep), each holding a
// block column's pixels for the tile: the pixel rows around one block row, 4
// (2 values each) or 3 (low-res, 1 each). Three are the window's left, centre
// and right columns; the fourth is filled for the next period, group by group
// in the first IN/PI steps, from the two line buffers and the new input. The
// slots' roles rotate with the period, and each memory has one write and one
// read at fixed places (a run-time choice of memory would duplicate them).
// The centre trails the input by a block row and two blocks, across row and
// patch boundaries, so the image's blocks follow one another without gaps:
// n x blocks + W + 2 periods for n patches. Taps outside the image hold other
// rows' or patches' values: the centre's position zeroes them.

#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
typedef ap_int<18> wv_t;   // a transformed input
typedef ap_int<12> wu_t;   // a transformed weight (x 4)
typedef ap_int<30> wp_t;   // their product
typedef ap_int<34> ws_t;   // a sum of products, modulo 2^34
#else
typedef int32_t wv_t;      // plain g++: wide enough for the same values,
typedef int16_t wu_t;
typedef int32_t wp_t;
typedef int64_t ws_t;      // and the sums exact (the low 32 bits of 4Y >> 2 are the same)
#endif

// The engine's shape. UP: an UpBlock (low-res pixels in).
template<int IN, int OUT, int PI, int PO, bool UP>
struct wino_shape {
    static_assert(IN % PI == 0 && OUT % PO == 0, "PI must divide IN, PO must divide OUT");
    static constexpr int IG = IN / PI, OG = OUT / PO, STEPS = IG * OG;
    static constexpr int NV = UP ? 9 : 16;      // multiplies per input x output channel
    static constexpr int PX = UP ? 1 : 4;       // pixels per input vector
    static constexpr int SLOT = UP ? 3 : 8;     // a column's values per channel in the window
    static constexpr int LA = UP ? 1 : 2;       // line buffer A: the pixel row above the block row
    static constexpr int LB = UP ? 1 : 4;       // line buffer B: the block row (or pixel row) itself
    static constexpr int LANES = PI * PO * NV;  // weights per step
};

// V's elements that are used: all 16, or (up) rows and columns 0, 1, 3.
constexpr int wino_row(bool up, int k) { return up ? (k / 3 == 2 ? 3 : k / 3) : k / 4; }
constexpr int wino_col(bool up, int k) { return up ? (k % 3 == 2 ? 3 : k % 3) : k % 4; }

// V = B^T d B, B^T = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]].
inline void wino_in_tile(const act_t d[4][4], wv_t v[4][4]) {
    #pragma HLS INLINE
    int32_t t[4][4];
    for (int j = 0; j < 4; j++) {
        #pragma HLS UNROLL
        t[0][j] = d[0][j] - d[2][j];
        t[1][j] = d[1][j] + d[2][j];
        t[2][j] = d[2][j] - d[1][j];
        t[3][j] = d[1][j] - d[3][j];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        v[i][0] = t[i][0] - t[i][2];
        v[i][1] = t[i][1] + t[i][2];
        v[i][2] = t[i][2] - t[i][1];
        v[i][3] = t[i][1] - t[i][3];
    }
}

// The same for the upsampled tile of a low-res 3x3 neighbourhood n: B^T on
// (a, b, b, c) is (a - b, 2b, 0, b - c). Row and column 2 are zero (never used).
inline void wino_in_up(const act_t n[3][3], wv_t v[4][4]) {
    #pragma HLS INLINE
    int32_t t[4][3];
    for (int j = 0; j < 3; j++) {
        #pragma HLS UNROLL
        t[0][j] = n[0][j] - n[1][j];
        t[1][j] = 2 * n[1][j];
        t[2][j] = 0;
        t[3][j] = n[1][j] - n[2][j];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        v[i][0] = t[i][0] - t[i][1];
        v[i][1] = 2 * t[i][1];
        v[i][2] = 0;
        v[i][3] = t[i][1] - t[i][2];
    }
}

// 4Y = A^T M A, A^T = [[1, 1, 1, 0], [0, 1, -1, -1]].
inline void wino_out(const ws_t m[4][4], ws_t y[2][2]) {
    #pragma HLS INLINE
    ws_t t[2][4];
    for (int j = 0; j < 4; j++) {
        #pragma HLS UNROLL
        t[0][j] = m[0][j] + m[1][j] + m[2][j];
        t[1][j] = m[1][j] - m[2][j] - m[3][j];
    }
    for (int i = 0; i < 2; i++) {
        #pragma HLS UNROLL
        y[i][0] = t[i][0] + t[i][1] + t[i][2];
        y[i][1] = t[i][1] - t[i][2] - t[i][3];
    }
}

// U = (2G) g (2G)^T, 2G = [[2, 0, 0], [1, 1, 1], [1, -1, 1], [0, 0, 2]].
inline void wino_weight(const wgt_t g[3][3], int32_t u[4][4]) {
    #pragma HLS INLINE
    int32_t t[4][3];
    for (int b = 0; b < 3; b++) {
        #pragma HLS UNROLL
        t[0][b] = 2 * g[0][b];
        t[1][b] = g[0][b] + g[1][b] + g[2][b];
        t[2][b] = g[0][b] - g[1][b] + g[2][b];
        t[3][b] = 2 * g[2][b];
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        u[i][0] = 2 * t[i][0];
        u[i][1] = t[i][0] + t[i][1] + t[i][2];
        u[i][2] = t[i][0] - t[i][1] + t[i][2];
        u[i][3] = 2 * t[i][2];
    }
}

// One step's weights: [p][j][k], PI x PO x NV, one wide memory word.
template<int LANES>
struct wino_word_t {
    wu_t v[LANES];
};

// The layer's weights, transformed, in the order the steps use them: word st =
// og * IG + ig. The biases shifted left to the sums' scale, [og][j].
template<int IN, int OUT, int PI, int PO, bool UP, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT]>
void load_wino_weights(wino_word_t<wino_shape<IN, OUT, PI, PO, UP>::LANES> wu[wino_shape<IN, OUT, PI, PO, UP>::STEPS],
                       acc_t bl[OUT / PO][PO]) {
    typedef wino_shape<IN, OUT, PI, PO, UP> S;
    // A step's word at once: its PI x PO x 9 weights from the ROM, as many
    // cycles as its ports take (HLS picks the II), once, when the layer starts.
    load_w: for (int st = 0; st < S::STEPS; st++) {
        #pragma HLS PIPELINE
        const int og = st / S::IG, ig = st % S::IG;
        wino_word_t<S::LANES> word;
        for (int p = 0; p < PI; p++) {
            #pragma HLS UNROLL
            for (int j = 0; j < PO; j++) {
                #pragma HLS UNROLL
                int32_t u[4][4];
                wino_weight(WEIGHT[og * PO + j][ig * PI + p], u);
                for (int k = 0; k < S::NV; k++) {
                    #pragma HLS UNROLL
                    word.v[(p * PO + j) * S::NV + k] = u[wino_row(UP, k)][wino_col(UP, k)];
                }
            }
        }
        wu[st] = word;
    }
    load_b: for (int oc = 0; oc < OUT; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc / PO][oc % PO] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
    }
}

// The engine, on n patches of GH x GW centres (blocks, or low-res pixels). In:
// a vector of PX pixels x IN channels per centre; out: a block of 4 pixels x
// OUT, ReLU'd and rounded off SHIFT bits to int16 activations.
template<int GH, int GW, int IN, int OUT, int PI, int PO, bool UP, int SHIFT, bool DSP>
void wino_run(hls::stream<vec_t<act_t, wino_shape<IN, OUT, PI, PO, UP>::PX * IN> >& in,
              const wino_word_t<wino_shape<IN, OUT, PI, PO, UP>::LANES> wu[wino_shape<IN, OUT, PI, PO, UP>::STEPS],
              const acc_t bl[OUT / PO][PO],
              hls::stream<vec_t<act_t, 4 * OUT> >& out, int n) {
    typedef wino_shape<IN, OUT, PI, PO, UP> S;
    constexpr int IG = S::IG, OG = S::OG, NV = S::NV, PX = S::PX;
    constexpr int CENTRES = GH * GW, LAG = GW + 2;
    struct col_t { act_t v[PI][S::SLOT]; };     // a window column, PI channels
    struct la_t { act_t v[PI][S::LA]; };
    struct lb_t { act_t v[PI][S::LB]; };

    col_t slot[4][IG];          // the column slots, [slot][input group]
    #pragma HLS ARRAY_PARTITION variable=slot type=complete dim=1
    #pragma HLS AGGREGATE variable=slot compact=bit
    la_t line_a[GW][IG];        // per column: the pixel row above the block row being filled
    lb_t line_b[GW][IG];        // per column: the block row (pixel row) being filled
    #pragma HLS AGGREGATE variable=line_a compact=bit
    #pragma HLS AGGREGATE variable=line_b compact=bit
    ws_t m[PO][4][4];           // the accumulators of this output group
    #pragma HLS ARRAY_PARTITION variable=m type=complete dim=0
    act_t blk[PX * IN];         // this period's input
    #pragma HLS ARRAY_PARTITION variable=blk type=complete
    act_t ob[4 * OUT];          // this period's output, filled group by group
    #pragma HLS ARRAY_PARTITION variable=ob type=complete

    // Defined contents, though only taps the centre's position zeroes ever read
    // them before they are written.
    clear: for (int k = 0; k < GW * IG; k++) {
        #pragma HLS PIPELINE II=1
        la_t a;
        lb_t c;
        for (int p = 0; p < PI; p++) {
            #pragma HLS UNROLL
            for (int q = 0; q < S::LA; q++) a.v[p][q] = 0;
            for (int q = 0; q < S::LB; q++) c.v[p][q] = 0;
        }
        line_a[k / IG][k % IG] = a;
        line_b[k / IG][k % IG] = c;
    }

    const int total = n * CENTRES;
    int b = 0;                  // period: input b comes in, column b - GW is filled, centre b - LAG done
    int og = 0, ig = 0;         // this step's output and input group
    int ix = 0;                 // input b's column
    int ring = (4 - GW % 4) % 4;// the slot filled this period: (b - GW) mod 4
    int cy = 0, cx = 0;         // the centre's position in its patch

    constexpr int TRIPS = (CENTRES + LAG) * IG * OG, MAX_TRIPS = (665 * CENTRES + LAG) * IG * OG;
    (void)TRIPS; (void)MAX_TRIPS;   // only the pragma uses them
    conv: for (int t = 0; t < (total + LAG) * IG * OG; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=TRIPS max=MAX_TRIPS
        // A slot is written one period before it is read, and rewritten four
        // periods later; a line buffer column is read and written once a period,
        // in that order, at the same group.
        #pragma HLS DEPENDENCE variable=slot type=inter false
        #pragma HLS DEPENDENCE variable=line_a type=inter false
        #pragma HLS DEPENDENCE variable=line_b type=inter false
        const bool fill = og == 0;              // the first IG steps fill the next column
        const bool live = b >= LAG && b < total + LAG;

        if (fill && ig == 0) {
            const bool have = b < total;
            vec_t<act_t, PX * IN> x;
            if (have) x = in.read();
            for (int i = 0; i < PX * IN; i++) {
                #pragma HLS UNROLL
                blk[i] = have ? x.v[i] : (act_t)0;
            }
        }

        if (fill) {
            // Column b - GW: the rows above (line buffers) and below (the input).
            const la_t a = line_a[ix][ig];
            const lb_t c = line_b[ix][ig];
            col_t col;
            la_t na;
            lb_t nb;
            for (int p = 0; p < PI; p++) {
                #pragma HLS UNROLL
                const int ch = ig * PI + p;
                if (UP) {
                    col.v[p][0] = a.v[p][0];
                    col.v[p][1] = c.v[p][0];
                    col.v[p][2] = blk[ch];
                    na.v[p][0] = c.v[p][0];
                    nb.v[p][0] = blk[ch];
                } else {
                    // [row of the tile][dx]: the pixel row above, the block, the row below.
                    for (int dx = 0; dx < 2; dx++) {
                        #pragma HLS UNROLL
                        col.v[p][dx] = a.v[p][dx];
                        col.v[p][2 + dx] = c.v[p][dx];
                        col.v[p][4 + dx] = c.v[p][2 + dx];
                        col.v[p][6 + dx] = blk[dx * IN + ch];
                        na.v[p][dx] = c.v[p][2 + dx];
                    }
                    for (int q = 0; q < 4; q++) {
                        #pragma HLS UNROLL
                        nb.v[p][q] = blk[q * IN + ch];
                    }
                }
            }
            line_a[ix][ig] = na;
            line_b[ix][ig] = nb;
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                if (ring == k) slot[k][ig] = col;
            }
        }

        // The window: left, centre and right columns of input group ig.
        col_t rd[4];
        #pragma HLS ARRAY_PARTITION variable=rd type=complete
        for (int k = 0; k < 4; k++) {
            #pragma HLS UNROLL
            rd[k] = slot[k][ig];
        }
        const col_t &left = rd[(ring + 1) & 3], &centre = rd[(ring + 2) & 3], &right = rd[(ring + 3) & 3];
        const bool up = cy > 0, down = cy < GH - 1, before = cx > 0, after = cx < GW - 1;

        wv_t v[PI][4][4];
        #pragma HLS ARRAY_PARTITION variable=v type=complete dim=0
        for (int p = 0; p < PI; p++) {
            #pragma HLS UNROLL
            if (UP) {
                act_t nb3[3][3];
                for (int r = 0; r < 3; r++) {
                    #pragma HLS UNROLL
                    const bool row_ok = r == 0 ? up : r == 2 ? down : true;
                    nb3[r][0] = row_ok && before ? left.v[p][r] : (act_t)0;
                    nb3[r][1] = row_ok ? centre.v[p][r] : (act_t)0;
                    nb3[r][2] = row_ok && after ? right.v[p][r] : (act_t)0;
                }
                wino_in_up(nb3, v[p]);
            } else {
                act_t d[4][4];
                for (int r = 0; r < 4; r++) {
                    #pragma HLS UNROLL
                    const bool row_ok = r == 0 ? up : r == 3 ? down : true;
                    d[r][0] = row_ok && before ? left.v[p][2 * r + 1] : (act_t)0;
                    d[r][1] = row_ok ? centre.v[p][2 * r] : (act_t)0;
                    d[r][2] = row_ok ? centre.v[p][2 * r + 1] : (act_t)0;
                    d[r][3] = row_ok && after ? right.v[p][2 * r] : (act_t)0;
                }
                wino_in_tile(d, v[p]);
            }
        }

        const wino_word_t<S::LANES> w = wu[og * IG + ig];
        for (int j = 0; j < PO; j++) {
            #pragma HLS UNROLL
            for (int k = 0; k < NV; k++) {
                #pragma HLS UNROLL
                const int r = wino_row(UP, k), c = wino_col(UP, k);
                ws_t sum = 0;
                for (int p = 0; p < PI; p++) {
                    #pragma HLS UNROLL
                    sum += mul<DSP, wp_t>(v[p][r][c], w.v[(p * PO + j) * NV + k]);
                }
                m[j][r][c] = (ig == 0 ? (ws_t)0 : m[j][r][c]) + sum;
            }
            if (UP) {
                for (int k = 0; k < 4; k++) {
                    #pragma HLS UNROLL
                    m[j][2][k] = 0;
                    m[j][k][2] = 0;
                }
            }
            if (ig == IG - 1) {
                ws_t y[2][2];
                wino_out(m[j], y);
                for (int q = 0; q < 4; q++) {
                    #pragma HLS UNROLL
                    const acc_t sum = (acc_t)(y[q / 2][q % 2] >> 2) + bl[og][j];
                    for (int g = 0; g < OG; g++) {
                        #pragma HLS UNROLL
                        if (og == g) ob[q * OUT + g * PO + j] = requant_relu<SHIFT>(sum);
                    }
                }
            }
        }
        if (live && og == OG - 1 && ig == IG - 1) {
            vec_t<act_t, 4 * OUT> o;
            for (int i = 0; i < 4 * OUT; i++) {
                #pragma HLS UNROLL
                o.v[i] = ob[i];
            }
            out.write(o);
        }

        if (ig < IG - 1) {
            ig++;
        } else {
            ig = 0;
            if (og < OG - 1) {
                og++;
            } else {
                og = 0;
                if (live) {
                    if (cx < GW - 1) cx++;
                    else { cx = 0; cy = cy < GH - 1 ? cy + 1 : 0; }
                }
                ix = ix < GW - 1 ? ix + 1 : 0;
                ring = (ring + 1) & 3;
                b++;
            }
        }
    }
}

// The weights live in a wide memory, one word a step: block RAM when it is
// deep, LUTs (distributed RAM) when it is shallow, where block RAM would hold
// a few rows of a mostly empty width. HLS's impl takes only a literal word:
// one branch each, HLS keeps the one STEPS picks.
template<int PH, int PW, int GH, int GW, int IN, int OUT, int PI, int PO, bool UP, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT], bool DSP>
void wino_layer(hls::stream<vec_t<act_t, wino_shape<IN, OUT, PI, PO, UP>::PX * IN> >& in,
                hls::stream<vec_t<act_t, 4 * OUT> >& out, int height, int width) {
    #pragma HLS INLINE
    typedef wino_shape<IN, OUT, PI, PO, UP> S;
    acc_t bl[OUT / PO][PO];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete dim=2
    const int n = patch_count<PH, PW>(height, width);
    if (S::STEPS > 64) {
        wino_word_t<S::LANES> wu[S::STEPS];
        #pragma HLS AGGREGATE variable=wu compact=bit
        #pragma HLS BIND_STORAGE variable=wu type=ram_2p impl=bram
        load_wino_weights<IN, OUT, PI, PO, UP, BIAS_SHIFT, WEIGHT, BIAS>(wu, bl);
        wino_run<GH, GW, IN, OUT, PI, PO, UP, SHIFT, DSP>(in, wu, bl, out, n);
    } else {
        wino_word_t<S::LANES> wu[S::STEPS];
        #pragma HLS AGGREGATE variable=wu compact=bit
        #pragma HLS BIND_STORAGE variable=wu type=ram_2p impl=lutram
        load_wino_weights<IN, OUT, PI, PO, UP, BIAS_SHIFT, WEIGHT, BIAS>(wu, bl);
        wino_run<GH, GW, IN, OUT, PI, PO, UP, SHIFT, DSP>(in, wu, bl, out, n);
    }
}

// A 3x3 conv layer and its ReLU on an H x W map of each PH x PW patch
// (patch_grid.hpp): blocks in, blocks out. PI input x PO output channels a
// step, 16 x PI x PO multipliers (DSP: in DSPs, or LUTs), IN/PI x OUT/PO
// cycles a block, H/2 x W/2 blocks a patch. WEIGHT and BIAS: the layer's int8
// arrays, template arguments so that they stay a ROM inside this process.
// BIAS_SHIFT: the sums' fraction bits minus the bias's; SHIFT: the sums' minus
// the activations'.
template<int PH, int PW, int H, int W, int IN, int OUT, int PI, int PO, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT], bool DSP>
void wino_conv3x3(hls::stream<vec_t<act_t, 4 * IN> >& in, hls::stream<vec_t<act_t, 4 * OUT> >& out,
                  int height, int width) {
    static_assert(H % 2 == 0 && W % 2 == 0, "whole 2x2 blocks");
    wino_layer<PH, PW, H / 2, W / 2, IN, OUT, PI, PO, false, BIAS_SHIFT, SHIFT, WEIGHT, BIAS, DSP>(
        in, out, height, width);
}

// An UpBlock and its ReLU: an H x W map (raster order, one vector a pixel),
// upsampled x2 by repeating each pixel, then the 3x3 conv: blocks of the 2H x
// 2W map out, one per input pixel, so in raster order of its block grid. 9 x
// PI x PO multipliers, IN/PI x OUT/PO cycles a pixel.
template<int PH, int PW, int H, int W, int IN, int OUT, int PI, int PO, int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT], bool DSP>
void wino_up_conv3x3(hls::stream<vec_t<act_t, IN> >& in, hls::stream<vec_t<act_t, 4 * OUT> >& out,
                     int height, int width) {
    wino_layer<PH, PW, H, W, IN, OUT, PI, PO, true, BIAS_SHIFT, SHIFT, WEIGHT, BIAS, DSP>(
        in, out, height, width);
}
