#pragma once

#include "fixed_point.hpp"
#include "window3x3.hpp"
#include "patch_grid.hpp"

// Copies a conv's weights from ROM into a local array the caller partitions, and
// its biases shifted up to the sums' fraction bits.
template<int OUT_CHANNELS, int IN_CHANNELS, int BIAS_SHIFT>
void load_conv_weights(const weight_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t bias[OUT_CHANNELS],
                       weight_t local_weight[OUT_CHANNELS][IN_CHANNELS][3][3], sum_t local_bias[OUT_CHANNELS]) {
    load_w: for (int out = 0; out < OUT_CHANNELS; out++)
        for (int in = 0; in < IN_CHANNELS; in++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    #pragma HLS PIPELINE II=1
                    local_weight[out][in][ky][kx] = weight[out][in][ky][kx];
                }
    load_b: for (int out = 0; out < OUT_CHANNELS; out++) {
        #pragma HLS PIPELINE II=1
        local_bias[out] = (sum_t)bias[out] * ((sum_t)1 << BIAS_SHIFT);
    }
}

// A same-padded 3x3 conv of one map, streamed: every input channel of a pixel
// in, raster order; every pixel's outputs out (conv_out: sums or activations),
// the same order. Two lines of input
// are kept on chip, the window in registers.
//
// The window's centre trails the input by a row and a pixel, so the loop runs
// W + 1 steps past the last input. Taps outside the map are zeroed by the
// centre's position. OUT_PER_CYCLE output channels a cycle, so a layer with
// time to spare uses fewer multipliers.
template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int OUT_PER_CYCLE,
         activation A, int ROUND_BITS, bool DSP, typename A_T>
void conv3x3_all_in_patch(hls::stream<vec_t<A_T, IN_CHANNELS> >& in,
                          const weight_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], const sum_t bias[OUT_CHANNELS],
                          hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out) {
    constexpr int STEPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;

    A_T lines[2][MAP_WIDTH][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    A_T window[3][3][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    sum_t sums[OUT_CHANNELS];
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
        const int first_out = STEPS == 1 ? 0 : step * OUT_PER_CYCLE;

        if (first_step) {
            const bool have = inputs < PIXELS;
            vec_t<A_T, IN_CHANNELS> pixel;
            if (have) pixel = in.read();
            shift_window<MAP_WIDTH, IN_CHANNELS>(lines, window, in_col, pixel, have);
        }

        if (inputs >= LAG) {
            const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
            const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
            for (int j = 0; j < OUT_PER_CYCLE; j++) {
                #pragma HLS UNROLL
                const int o = first_out + j;
                sum_t sum = 0;
                for (int ch = 0; ch < IN_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    for (int ky = 0; ky < 3; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            const A_T a = (row_inside[ky] && col_inside[kx]) ? window[ky][kx][ch] : (A_T)0;
                            sum += mul<DSP, sum_t>(a, weight[o][ch][ky][kx]);
                        }
                    }
                }
                sums[o] = bias[o] + sum;
            }
            if (last_step) {
                vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> pixel_out;
                for (int o = 0; o < OUT_CHANNELS; o++) {
                    #pragma HLS UNROLL
                    pixel_out.v[o] = conv_out<A, ROUND_BITS>::of(sums[o]);
                }
                out.write(pixel_out);
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

// A 3x3 conv with few channels: every input channel at once, OUT_PER_CYCLE output
// channels a cycle. A: its activation (conv_out), ROUND_BITS what that rounds off.
// The weights stay a ROM inside this process.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int BIAS_SHIFT, const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS],
         bool DSP, int OUT_PER_CYCLE = 1, activation A = activation::none, int ROUND_BITS = 0>
void conv3x3_all_in(hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
                    hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out,
                    int height, int width) {
    weight_t weight[OUT_CHANNELS][IN_CHANNELS][3][3];
    #pragma HLS ARRAY_PARTITION variable=weight type=cyclic factor=OUT_PER_CYCLE dim=1
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=4
    sum_t bias[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bias type=cyclic factor=OUT_PER_CYCLE dim=1

    load_conv_weights<OUT_CHANNELS, IN_CHANNELS, BIAS_SHIFT>(WEIGHT, BIAS, weight, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_all_in_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, OUT_PER_CYCLE,
                             A, ROUND_BITS, DSP>(in, weight, bias, out);
    }
}
