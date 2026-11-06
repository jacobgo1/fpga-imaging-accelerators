#pragma once

#include "fixed_point.hpp"
#include "window3x3.hpp"
#include "patch_grid.hpp"

// A same-padded 3x3 conv of one map, streamed: a pixel with all its channels in,
// raster order; its outputs out, the same order. Two lines of input are kept on
// chip, the window in registers; its centre trails the input by a row and a
// pixel, so the loop runs W + 1 pixels past the last input, and taps outside the
// map are zeroed by the centre's position.
//
// Sized for any time a pixel: a step is the whole window for IN_PER_CYCLE input
// channels times OUT_PER_CYCLE output channels (IN_PER_CYCLE = IN_CHANNELS for a
// small layer); a pixel takes OUT_GROUPS x IN_GROUPS steps, output group outer.
// Out the sums, or with activation::relu ReLU'd and rounded off ROUND_BITS bits
// to activations, so no ReLU module is needed after it.
//
// The weights are one wide memory word a step: one memory a layer, since a
// thousand small ones made sp_unet_large_opt too congested to route.
template<int IN_PER_CYCLE, int OUT_PER_CYCLE>
struct tiled_word_t {
    weight_t v[OUT_PER_CYCLE][IN_PER_CYCLE][3][3];
};

template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE,
         activation A, int ROUND_BITS, bool DSP>
void conv3x3_tiled_patch(
    hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
    const tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE)],
    const sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE],
    hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;
    static_assert((MAP_WIDTH & (MAP_WIDTH - 1)) == 0, "a power of two: the centre's y and x are shifts");
    typedef typename conv_out<A, ROUND_BITS>::type out_t;

    act_t lines[2][MAP_WIDTH][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    act_t window[3][3][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    out_t outputs[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=outputs type=complete
    sum_t running[OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=running type=complete
    int in_col = 0;

    // A perfect loop nest, pipelined at the bottom: HLS flattens it into one
    // pipeline for the patch, so a pixel of few steps costs no flush.
    pixels: for (int p = 0; p < PIXELS + LAG; p++)
        out_groups: for (int og = 0; og < OUT_GROUPS; og++)
            in_groups: for (int ig = 0; ig < IN_GROUPS; ig++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=lines type=inter false
                // A pixel's first step takes input pixel p in; past the last, zeros.
                if (og == 0 && ig == 0) {
                    const bool have = p < PIXELS;
                    vec_t<act_t, IN_CHANNELS> value;
                    if (have) value = in.read();
                    shift_window<MAP_WIDTH, IN_CHANNELS>(lines, window, in_col, value, have);
                }

                // The window's centre is pixel p - LAG.
                if (p >= LAG) {
                    const unsigned centre = p - LAG;
                    const int y = centre / MAP_WIDTH, x = centre % MAP_WIDTH;
                    const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
                    const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
                    const tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> word = weight[og * IN_GROUPS + ig];
                    for (int j = 0; j < OUT_PER_CYCLE; j++) {
                        #pragma HLS UNROLL
                        sum_t sum = 0;
                        for (int i = 0; i < IN_PER_CYCLE; i++) {
                            #pragma HLS UNROLL
                            for (int ky = 0; ky < 3; ky++) {
                                #pragma HLS UNROLL
                                for (int kx = 0; kx < 3; kx++) {
                                    #pragma HLS UNROLL
                                    const act_t a = (row_inside[ky] && col_inside[kx])
                                                  ? window[ky][kx][ig * IN_PER_CYCLE + i] : (act_t)0;
                                    sum += mul<DSP, sum_t>(a, word.v[j][i][ky][kx]);
                                }
                            }
                        }
                        const sum_t total = (ig == 0 ? bias[og][j] : running[j]) + sum;
                        running[j] = total;
                        // Rounded as its output group finishes: OUT_PER_CYCLE rounding units, not OUT_CHANNELS.
                        const out_t done = conv_out<A, ROUND_BITS>::of(total);
                        for (int g = 0; g < OUT_GROUPS; g++) {
                            #pragma HLS UNROLL
                            if (ig == IN_GROUPS - 1 && og == g) outputs[g * OUT_PER_CYCLE + j] = done;
                        }
                    }
                    if (og == OUT_GROUPS - 1 && ig == IN_GROUPS - 1) {
                        vec_t<out_t, OUT_CHANNELS> pixel_out;
                        for (int c = 0; c < OUT_CHANNELS; c++) {
                            #pragma HLS UNROLL
                            pixel_out.v[c] = outputs[c];
                        }
                        out.write(pixel_out);
                    }
                }
            }
}

// The weights a word a step, in the order the steps take them, read from the
// ROM once, when the layer starts.
template<int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS]>
void load_tiled_weights(
    tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE)],
    sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE]) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    load_w: for (int og = 0; og < OUT_GROUPS; og++)
        for (int ig = 0; ig < IN_GROUPS; ig++) {
            #pragma HLS PIPELINE
            tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> word;
            for (int j = 0; j < OUT_PER_CYCLE; j++) {
                #pragma HLS UNROLL
                for (int i = 0; i < IN_PER_CYCLE; i++) {
                    #pragma HLS UNROLL
                    for (int ky = 0; ky < 3; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            word.v[j][i][ky][kx] = WEIGHT[og * OUT_PER_CYCLE + j][ig * IN_PER_CYCLE + i][ky][kx];
                        }
                    }
                }
            }
            weight[og * IN_GROUPS + ig] = word;
        }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o / OUT_PER_CYCLE][o % OUT_PER_CYCLE] = (sum_t)BIAS[o] * ((sum_t)1 << BIAS_SHIFT);
    }
}

// The layer once its weight memory is declared: the weights loaded, then every patch.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS], bool DSP,
         activation A, int ROUND_BITS>
void conv3x3_tiled_layer(
    tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE)],
    hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
    hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out, int height, int width) {
    #pragma HLS INLINE
    sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    load_tiled_weights<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, BIAS_SHIFT, WEIGHT, BIAS>(weight, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        conv3x3_tiled_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE,
                            A, ROUND_BITS, DSP>(in, weight, bias, out);
    }
}

// The weight memory is block RAM when deep, LUTs when shallow (block RAM would
// be mostly empty). impl takes only a literal word: one declaration each.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS], bool DSP,
         activation A = activation::none, int ROUND_BITS = 0>
void conv3x3_tiled(hls::stream<vec_t<act_t, IN_CHANNELS> >& in,
                   hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out,
                   int height, int width) {
    constexpr int STEPS = (OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE);
    if (STEPS > 64) {
        tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> weight[STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=bram
        conv3x3_tiled_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
                            OUT_PER_CYCLE, BIAS_SHIFT, WEIGHT, BIAS, DSP, A, ROUND_BITS>(weight, in, out, height, width);
    } else {
        tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> weight[STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=lutram
        conv3x3_tiled_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
                            OUT_PER_CYCLE, BIAS_SHIFT, WEIGHT, BIAS, DSP, A, ROUND_BITS>(weight, in, out, height, width);
    }
}
