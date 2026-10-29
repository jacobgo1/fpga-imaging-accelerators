#pragma once

#include "fixed_point.hpp"

// Copies a conv's weights from ROM into a local array the caller partitions, and
// its biases shifted up to the sums' fraction bits.
template<int OUT_CHANNELS, int IN_CHANNELS, int BIAS_SHIFT>
void load_conv_weights(const wgt_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t bias[OUT_CHANNELS],
                       wgt_t local_weight[OUT_CHANNELS][IN_CHANNELS][3][3], acc_t local_bias[OUT_CHANNELS]) {
    load_w: for (int out = 0; out < OUT_CHANNELS; out++)
        for (int in = 0; in < IN_CHANNELS; in++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    #pragma HLS PIPELINE II=1
                    local_weight[out][in][ky][kx] = weight[out][in][ky][kx];
                }
    load_b: for (int out = 0; out < OUT_CHANNELS; out++) {
        #pragma HLS PIPELINE II=1
        local_bias[out] = (acc_t)bias[out] * ((acc_t)1 << BIAS_SHIFT);
    }
}

// A same-padded 3x3 conv of one map, streamed: every input channel of a pixel
// in, raster order; every pixel's sums out, the same order. Two lines of input
// are kept on chip, the window in registers.
//
// The window's centre trails the input by a row and a pixel, so the loop runs
// W + 1 steps past the last input. Taps outside the map are zeroed by the
// centre's position. STEPS spreads the output channels over STEPS cycles a
// pixel, so a layer with time to spare uses fewer multipliers.
template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int STEPS, bool DSP, typename A_T>
void conv3x3_stream(hls::stream<vec_t<A_T, IN_CHANNELS> >& in,
                    const wgt_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], const acc_t bias[OUT_CHANNELS],
                    hls::stream<vec_t<acc_t, OUT_CHANNELS> >& out) {
    constexpr int OUT_PER_STEP = OUT_CHANNELS / STEPS;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;

    A_T lines[2][MAP_WIDTH][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    A_T window[3][3][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    acc_t sums[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete

    int in_col = 0;
    int y = 0, x = 0;               // the window's centre
    int step = 0, inputs = 0;       // inputs: input steps so far, the last W + 1 without input

    conv: for (int t = 0; t < (PIXELS + LAG) * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=lines type=inter false
        // Spelled out so that with one step they are constants.
        const bool first_step = STEPS == 1 || step == 0;
        const bool last_step = STEPS == 1 || step == STEPS - 1;
        const int first_out = STEPS == 1 ? 0 : step * OUT_PER_STEP;

        if (first_step) {
            // A new window column: rows r - 2, r - 1, r at column in_col.
            const bool have = inputs < PIXELS;
            vec_t<A_T, IN_CHANNELS> pixel;
            if (have) pixel = in.read();
            for (int ch = 0; ch < IN_CHANNELS; ch++) {
                #pragma HLS UNROLL
                const A_T two_up = lines[0][in_col][ch], one_up = lines[1][in_col][ch];
                const A_T now = have ? pixel.v[ch] : (A_T)0;
                lines[0][in_col][ch] = one_up;
                lines[1][in_col][ch] = now;
                for (int ky = 0; ky < 3; ky++) {
                    #pragma HLS UNROLL
                    window[ky][0][ch] = window[ky][1][ch];
                    window[ky][1][ch] = window[ky][2][ch];
                }
                window[0][2][ch] = two_up;
                window[1][2][ch] = one_up;
                window[2][2][ch] = now;
            }
            in_col = (in_col == MAP_WIDTH - 1) ? 0 : in_col + 1;
        }

        if (inputs >= LAG) {
            const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
            const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
            for (int j = 0; j < OUT_PER_STEP; j++) {
                #pragma HLS UNROLL
                const int o = first_out + j;
                acc_t sum = 0;
                for (int ch = 0; ch < IN_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    for (int ky = 0; ky < 3; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            const A_T a = (row_inside[ky] && col_inside[kx]) ? window[ky][kx][ch] : (A_T)0;
                            sum += mul<DSP, acc_t>(a, weight[o][ch][ky][kx]);
                        }
                    }
                }
                sums[o] = bias[o] + sum;
            }
            if (last_step) {
                vec_t<acc_t, OUT_CHANNELS> pixel_sums;
                for (int o = 0; o < OUT_CHANNELS; o++) {
                    #pragma HLS UNROLL
                    pixel_sums.v[o] = sums[o];
                }
                out.write(pixel_sums);
            }
        }

        if (last_step) {
            step = 0;
            if (inputs >= LAG) {
                if (x < MAP_WIDTH - 1) x++;
                else if (y < MAP_HEIGHT - 1) { x = 0; y++; }
                else { x = 0; y = 0; }
            }
            inputs++;
        } else {
            step++;
        }
    }
}
