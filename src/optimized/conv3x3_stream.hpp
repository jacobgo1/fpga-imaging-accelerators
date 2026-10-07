#pragma once

#include <stdint.h>
#include "../common/hls_stream_shim.hpp"

// Optimized building blocks for streaming CNN kernels, in integer fixed point.
// The golden/ templates say *what* is computed; these say *how fast*: they read
// their input from an hls::stream in the order they consume it, keep only two
// lines of it on chip, and do many multiply-adds per clock cycle.
//
// Numbers: an activation is an int16 a meaning a * 2^-F (F fraction bits, set
// by the kernel), a weight is the quantized int8 q meaning q * 2^-frac, so a
// product is exact in an int32 with F + frac fraction bits, and so is a sum of
// them. No floating point anywhere: a 16 x 8 bit multiply is one DSP, against
// about five for a float multiply-add.

typedef int16_t act_t;   // an activation (fixed point, F fraction bits)
typedef int8_t  wgt_t;   // a quantized weight q
typedef int32_t acc_t;   // a sum of products, and a bias at the same scale

// P values that travel together through a stream (P channels of one pixel).
template<typename T, int P>
struct vec_t {
    T v[P];
};

// Copy one conv layer's weights from ROM into a local array, and its biases
// shifted left to the scale of the sums (bias_shift = the sums' fraction bits
// minus the bias's). The caller partitions wl/bl as conv3x3_stream needs.
// IN_PAD > IN_CH pads the input channels with zero weights, so a layer can take
// IN_PAD channels at P per cycle when P does not divide IN_CH.
template<int OUT_CH, int IN_CH, int IN_PAD = IN_CH>
void load_conv_weights(const wgt_t w[OUT_CH][IN_CH][3][3], const wgt_t b[OUT_CH], int bias_shift,
                       wgt_t wl[OUT_CH][IN_PAD][3][3], acc_t bl[OUT_CH]) {
    static_assert(IN_PAD >= IN_CH, "IN_PAD must be at least IN_CH");
    load_w: for (int oc = 0; oc < OUT_CH; oc++)
        for (int ic = 0; ic < IN_PAD; ic++)
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++) {
                    #pragma HLS PIPELINE II=1
                    const int src = ic < IN_CH ? ic : IN_CH - 1;   // never read past w
                    wl[oc][ic][ki][kj] = ic < IN_CH ? w[oc][src][ki][kj] : (wgt_t)0;
                }
    load_b: for (int oc = 0; oc < OUT_CH; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc] = (acc_t)b[oc] * ((acc_t)1 << bias_shift);
    }
}

// Same-padded (zero) 3x3 convolution of one H x W image, streamed. Out: for
// every pixel, raster order, the OUT_CH sums bias + sum(act * q), unscaled.
//
// Input order: P channels at a time, plane by plane -- for each chunk c of P
// channels, every pixel in raster order (IN_CH/P * H * W vectors). With
// P = IN_CH that is simply one vector per pixel in raster order. A pixel's sums
// are complete in the last chunk, so they leave then, one pixel per step;
// earlier chunks' partial sums wait in acc.
//
// The 3x3 window's centre trails the input by one row and one pixel (W + 1
// vectors) all the way through, across row and chunk boundaries: the taps
// that fall outside the image are zeroed by the centre's position, not by
// feeding zeros in. So the loop runs IN_CH/P * H * W + W + 1 steps -- the
// only padding steps are the last W + 1, once per image.
//
// STEPS spreads the output channels over STEPS cycles per pixel: each cycle
// does OUT_CH/STEPS x P x 9 multiply-adds, so a layer with time to spare uses
// fewer multipliers. The weights of the other channels wait in small memories
// (wl partitioned complete in dims 2-4, not in dim 1, when STEPS > 1). With
// STEPS = 1 the caller partitions wl complete in dims 1, 3, 4 and cyclic by P
// in dim 2.
template<int H, int W, int IN_CH, int OUT_CH, int P, int STEPS>
void conv3x3_stream(hls::stream<vec_t<act_t, P> >& in,
                    const wgt_t wl[OUT_CH][IN_CH][3][3], const acc_t bl[OUT_CH],
                    hls::stream<vec_t<acc_t, OUT_CH> >& out) {
    static_assert(IN_CH % P == 0, "IN_CH must be a multiple of P");
    static_assert(OUT_CH % STEPS == 0, "OUT_CH must be a multiple of STEPS");
    constexpr int CHUNKS = IN_CH / P, PER_STEP = OUT_CH / STEPS;
    constexpr int VECTORS = CHUNKS * H * W, LAG = W + 1;

    // Partial sums between chunks (not needed, and 1 x 1, with one chunk).
    acc_t acc[CHUNKS > 1 ? H : 1][CHUNKS > 1 ? W : 1][OUT_CH];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete dim=3
    act_t lb[2][W][P];       // the two previous input lines
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=3
    act_t win[3][3][P];      // the 3x3 window, in registers
    #pragma HLS ARRAY_PARTITION variable=win type=complete dim=0
    acc_t pix[OUT_CH];       // the finished sums of one pixel, filled over STEPS
    #pragma HLS ARRAY_PARTITION variable=pix type=complete

    int s = 0;                  // column of the next input vector
    int co = 0, y = 0, x = 0;   // chunk and position of the window's centre
    int step = 0, g = 0;        // g: input steps so far, the last W + 1 without input

    conv: for (int t = 0; t < (VECTORS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        // acc[y][x] is written once per chunk and read H*W steps later; a line
        // buffer column, W steps later.
        #pragma HLS DEPENDENCE variable=acc type=inter false
        #pragma HLS DEPENDENCE variable=lb type=inter false
        // Spelled out so that with one chunk or one step they are constants
        // (HLS cannot prove that co or step never leave 0).
        const bool first_step = STEPS == 1 || step == 0;
        const bool last_step = STEPS == 1 || step == STEPS - 1;
        const bool first_chunk = CHUNKS == 1 || co == 0;
        const bool last_chunk = CHUNKS == 1 || co == CHUNKS - 1;
        const int chunk = CHUNKS == 1 ? 0 : co;      // the centre's chunk: its weights
        const int oc0 = STEPS == 1 ? 0 : step * PER_STEP;

        if (first_step) {
            // A new window column: input rows r-2, r-1, r at column s.
            const bool have = g < VECTORS;
            vec_t<act_t, P> v;
            if (have) v = in.read();
            for (int p = 0; p < P; p++) {
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
            // The window is centred on (y, x) of chunk co. Taps outside the
            // image hold other rows' or chunks' values: they count as zero.
            const bool row_ok[3] = {y > 0, true, y < H - 1};
            const bool col_ok[3] = {x > 0, true, x < W - 1};
            for (int j = 0; j < PER_STEP; j++) {
                #pragma HLS UNROLL
                const int oc = oc0 + j;
                acc_t sum = 0;
                for (int p = 0; p < P; p++) {
                    #pragma HLS UNROLL
                    for (int ki = 0; ki < 3; ki++) {
                        #pragma HLS UNROLL
                        for (int kj = 0; kj < 3; kj++) {
                            #pragma HLS UNROLL
                            const act_t a = (row_ok[ki] && col_ok[kj]) ? win[ki][kj][p] : (act_t)0;
                            sum += (acc_t)a * (acc_t)wl[oc][chunk * P + p][ki][kj];
                        }
                    }
                }
                const acc_t total = (first_chunk ? bl[oc] : acc[y][x][oc]) + sum;
                if (last_chunk) pix[oc] = total;
                else acc[y][x][oc] = total;
            }
            if (last_chunk && last_step) {
                vec_t<acc_t, OUT_CH> o;
                for (int oc = 0; oc < OUT_CH; oc++) {
                    #pragma HLS UNROLL
                    o.v[oc] = pix[oc];
                }
                out.write(o);
            }
        }

        if (last_step) {
            step = 0;
            if (g >= LAG) {
                if (x < W - 1) x++;
                else if (y < H - 1) { x = 0; y++; }
                else { x = 0; y = 0; co++; }
            }
            g++;
        } else {
            step++;
        }
    }
}
