#pragma once

#include "conv3x3_tiled.hpp"

// conv3x3_tiled, packed, for a first layer of many bands: a pixel comes in as
// IN_GROUPS chunks of IN_PER_CYCLE bands (what pixel_buffer sends), one a step, so
// no patch buffer is needed in front of it; raster order, the outputs the same.
// Channels past IN_CHANNELS in the last chunk get zero weights.
//
// Chunk g of a pixel enters the window at step g of the pixel's first output group,
// the step that first uses it. Every step is packed: two multiplies per DSP
// (dsp_pack.hpp), int8 inputs (-127 .. 127, groups of 8 on the DSP48E2) or uint8 (4).
template<typename IN_T> constexpr int input_max() { return std::is_same<IN_T, uact_t>::value ? 255 : 127; }

template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE,
         activation A, int ROUND_BITS, int PACK_SHIFT, typename IN_T>
void conv3x3_tiled_chunks_patch(
    hls::stream<vec_t<IN_T, IN_PER_CYCLE> >& in,
    const tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * ((IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE)],
    const sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE],
    hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out) {
    constexpr int IN_GROUPS = (IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int PADDED = IN_GROUPS * IN_PER_CYCLE;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;
    constexpr int TAPS = IN_PER_CYCLE * 9, GROUP = pack_group<PACK_SHIFT, input_max<IN_T>()>();
    static_assert((MAP_WIDTH & (MAP_WIDTH - 1)) == 0, "a power of two: the centre's y and x are shifts");
    static_assert(OUT_PER_CYCLE % 2 == 0, "whole pairs of output channels");
    static_assert(std::is_same<IN_T, uact_t>::value || std::is_same<IN_T, act8_t>::value, "8-bit inputs");
    typedef typename conv_out<A, ROUND_BITS>::type out_t;

    // The lines: a chunk's channels a memory each, the chunk in the address (a bank
    // chosen at run time would duplicate the stores).
    IN_T lines[2][IN_GROUPS * MAP_WIDTH][IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    IN_T window[3][3][PADDED];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    out_t outputs[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=outputs type=complete
    sum_t running[OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=running type=complete
    int in_col = 0;

    pixels: for (int p = 0; p < PIXELS + LAG; p++)
        out_groups: for (int og = 0; og < OUT_GROUPS; og++)
            in_groups: for (int ig = 0; ig < IN_GROUPS; ig++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=lines type=inter false
                // The first output group's steps take the pixel's chunks in; past the last pixel, zeros.
                if (og == 0) {
                    const bool have = p < PIXELS;
                    vec_t<IN_T, IN_PER_CYCLE> chunk;
                    if (have) chunk = in.read();
                    const int address = ig * MAP_WIDTH + in_col;
                    for (int c = 0; c < IN_PER_CYCLE; c++) {
                        #pragma HLS UNROLL
                        const IN_T two_up = lines[0][address][c], one_up = lines[1][address][c];
                        const IN_T now = have ? chunk.v[c] : (IN_T)0;
                        lines[0][address][c] = one_up;
                        lines[1][address][c] = now;
                        for (int g = 0; g < IN_GROUPS; g++) {
                            #pragma HLS UNROLL
                            if (g == ig) {
                                const int ch = g * IN_PER_CYCLE + c;
                                for (int ky = 0; ky < 3; ky++) {
                                    #pragma HLS UNROLL
                                    window[ky][0][ch] = window[ky][1][ch];
                                    window[ky][1][ch] = window[ky][2][ch];
                                }
                                window[0][2][ch] = two_up;
                                window[1][2][ch] = one_up;
                                window[2][2][ch] = now;
                            }
                        }
                    }
                    if (ig == IN_GROUPS - 1) in_col = (in_col == MAP_WIDTH - 1) ? 0 : in_col + 1;
                }

                // The window's centre is pixel p - LAG.
                if (p >= LAG) {
                    const unsigned centre = p - LAG;
                    const int y = centre / MAP_WIDTH, x = centre % MAP_WIDTH;
                    const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
                    const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
                    const tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT> word = weight[og * IN_GROUPS + ig];
                    for (int pair = 0; pair < OUT_PER_CYCLE / 2; pair++) {
                        #pragma HLS UNROLL
                        sum_t low = 0, high = 0;
                        for (int first = 0; first < TAPS; first += GROUP) {
                            #pragma HLS UNROLL
                            packed_sum_t sum = 0;
                            for (int t = first; t < first + GROUP && t < TAPS; t++) {
                                #pragma HLS UNROLL
                                const int i = t / 9, ky = (t % 9) / 3, kx = t % 3;
                                const IN_T a = (row_inside[ky] && col_inside[kx])
                                             ? window[ky][kx][ig * IN_PER_CYCLE + i] : (IN_T)0;
                                sum += packed_mul<PACK_SHIFT>(word.v[pair][i][ky][kx], a);
                            }
                            unpack_add<PACK_SHIFT>(sum, low, high);
                        }
                        tiled_finish<OUT_GROUPS, IN_GROUPS, OUT_PER_CYCLE, OUT_CHANNELS, A, ROUND_BITS>(
                            2 * pair, low, og, ig, bias, running, outputs);
                        tiled_finish<OUT_GROUPS, IN_GROUPS, OUT_PER_CYCLE, OUT_CHANNELS, A, ROUND_BITS>(
                            2 * pair + 1, high, og, ig, bias, running, outputs);
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

// The weights a word a step, packed at load (channel 2j + 1's above 2j's), zeros past
// IN_CHANNELS.
template<int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, int PACK_SHIFT, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS]>
void load_tiled_chunks_weights(
    tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * ((IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE)],
    sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE]) {
    constexpr int IN_GROUPS = (IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    load_w: for (int og = 0; og < OUT_GROUPS; og++)
        for (int ig = 0; ig < IN_GROUPS; ig++) {
            #pragma HLS PIPELINE
            tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT> word;
            for (int j = 0; j < OUT_PER_CYCLE / 2; j++) {
                #pragma HLS UNROLL
                for (int i = 0; i < IN_PER_CYCLE; i++) {
                    #pragma HLS UNROLL
                    for (int ky = 0; ky < 3; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            const int in_ch = ig * IN_PER_CYCLE + i, low = og * OUT_PER_CYCLE + 2 * j;
                            const bool real = in_ch < IN_CHANNELS;
                            const int source = real ? in_ch : IN_CHANNELS - 1;   // never read past the array
                            word.v[j][i][ky][kx] = pack_weights<PACK_SHIFT>(
                                real ? WEIGHT[low + 1][source][ky][kx] : (weight_t)0,
                                real ? WEIGHT[low][source][ky][kx] : (weight_t)0);
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

// The layer: its weights loaded once, then every patch. The weight memory as in
// conv3x3_tiled: block RAM when deep, LUTs when shallow.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS],
         int PACK_SHIFT, activation A, int ROUND_BITS, typename IN_T>
void conv3x3_tiled_chunks(hls::stream<vec_t<IN_T, IN_PER_CYCLE> >& in,
                          hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out,
                          int height, int width) {
    constexpr int STEPS = (OUT_CHANNELS / OUT_PER_CYCLE) * ((IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE);
    typedef tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT> word_t;
    word_t weight[STEPS];
    #pragma HLS AGGREGATE variable=weight compact=bit
    #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=lutram
    sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    static_assert(STEPS <= 64, "a deep weight memory belongs in block RAM: add that branch as in conv3x3_tiled");
    load_tiled_chunks_weights<IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT, BIAS_SHIFT,
                              WEIGHT, BIAS>(weight, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        conv3x3_tiled_chunks_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE,
                                   A, ROUND_BITS, PACK_SHIFT>(in, weight, bias, out);
    }
}
