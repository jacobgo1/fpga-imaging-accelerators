#pragma once

#include "conv3x3_stream.hpp"
#include "patch_grid.hpp"

// conv3x3_serial: a 3x3 conv layer for a chip with few DSPs: one input channel
// x one output channel per cycle, 9 multipliers whatever the layer's size,
// OUT x IN cycles per pixel. The same input and output as conv3x3_narrow (all
// channels of a pixel at once, raster order), so the two are interchangeable;
// this one is IN times slower and IN times smaller.
//
// The engine is conv3x3_stream's (read that first) with the steps over both
// channels: the window holds every input channel, and each step multiplies one
// of them by one output channel's 9 weights, adding to a running sum. The
// weights: 9 memories, one per tap, addressed by (output, input channel).
template<int H, int W, int IN_CH, int OUT_CH>
void conv3x3_serial_patch(hls::stream<vec_t<act_t, IN_CH> >& in,
                          const wgt_t wl[OUT_CH][IN_CH][3][3], const acc_t bl[OUT_CH],
                          hls::stream<vec_t<acc_t, OUT_CH> >& out) {
    constexpr int STEPS = OUT_CH * IN_CH;
    constexpr int VECTORS = H * W, LAG = W + 1;

    act_t lb[2][W][IN_CH];      // the two previous input lines
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=3
    act_t win[3][3][IN_CH];     // the 3x3 window, in registers
    #pragma HLS ARRAY_PARTITION variable=win type=complete dim=0
    acc_t pix[OUT_CH];          // the finished sums of one pixel
    #pragma HLS ARRAY_PARTITION variable=pix type=complete

    int s = 0;                  // column of the next input vector
    int y = 0, x = 0;           // position of the window's centre
    int oc = 0, ic = 0;         // this step's output and input channel
    int g = 0;                  // input steps so far, the last W + 1 without input
    acc_t run = 0;              // the sum of oc so far

    conv: for (int t = 0; t < (VECTORS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=lb type=inter false
        const bool first_step = oc == 0 && ic == 0;
        const bool last_in = ic == IN_CH - 1;
        const bool last_step = oc == OUT_CH - 1 && last_in;

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
            acc_t sum = 0;
            for (int ki = 0; ki < 3; ki++) {
                #pragma HLS UNROLL
                for (int kj = 0; kj < 3; kj++) {
                    #pragma HLS UNROLL
                    const act_t a = (row_ok[ki] && col_ok[kj]) ? win[ki][kj][ic] : (act_t)0;
                    sum += (acc_t)a * (acc_t)wl[oc][ic][ki][kj];
                }
            }
            const acc_t total = (ic == 0 ? bl[oc] : run) + sum;
            run = total;
            if (last_in) pix[oc] = total;
            if (last_step) {
                vec_t<acc_t, OUT_CH> o;
                for (int c = 0; c < OUT_CH; c++) {
                    #pragma HLS UNROLL
                    o.v[c] = pix[c];
                }
                out.write(o);
            }
        }

        if (!last_in) {
            ic++;
        } else {
            ic = 0;
            if (!last_step) {
                oc++;
            } else {
                oc = 0;
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

// The layer: the image is cut into PH x PW patches (patch_grid.hpp); the layer
// is H x W of each. WEIGHT and BIAS: the layer's quantized int8 arrays,
// template arguments so that they stay a constant ROM inside this process.
// BIAS_SHIFT: the sums' fraction bits minus the bias's.
template<int PH, int PW, int H, int W, int IN, int OUT, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT][IN][3][3], const wgt_t (&BIAS)[OUT]>
void conv3x3_serial(hls::stream<vec_t<act_t, IN> >& in, hls::stream<vec_t<acc_t, OUT> >& out,
                    int height, int width) {
    wgt_t wl[OUT][IN][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[OUT];

    load_conv_weights<OUT, IN>(WEIGHT, BIAS, BIAS_SHIFT, wl, bl);
    const int n = patch_count<PH, PW>(height, width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_serial_patch<H, W, IN, OUT>(in, wl, bl, out);
    }
}
