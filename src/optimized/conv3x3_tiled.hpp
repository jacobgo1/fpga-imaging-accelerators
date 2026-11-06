#pragma once

#include <type_traits>
#include "fixed_point.hpp"
#include "dsp_pack.hpp"
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
// IN_T: the input activations' type, int16 or uint8, multiplied as it is.
// PACKED: two multiplies per DSP (dsp_pack.hpp), uint8 inputs only. Output channels
// 2j and 2j + 1 of a step share every activation, so a pair's IN_PER_CYCLE x 9
// products go in groups of pack_group (4 on the DSP48E2), the last one shorter
// when they do not divide; each group's sum is split into the two channels.
// PACK_SHIFT: the chip's packing (the kernel's DSP_PACK_SHIFT).
//
// The weights are one wide memory word a step: one memory a layer, since a
// thousand small ones made sp_unet_large_opt too congested to route.
template<int IN_PER_CYCLE, int OUT_PER_CYCLE>
struct tiled_word_t {
    weight_t v[OUT_PER_CYCLE][IN_PER_CYCLE][3][3];
};

// Packed: channel 2j + 1's weight above 2j's, packed once at load, never next to the
// multiply (see dsp_pack.hpp).
template<int IN_PER_CYCLE, int OUT_PER_CYCLE, int SHIFT>
struct tiled_packed_word_t {
    packed_weight_t<SHIFT> v[OUT_PER_CYCLE / 2][IN_PER_CYCLE][3][3];
};

template<bool PACKED, int IN_PER_CYCLE, int OUT_PER_CYCLE, int SHIFT>
struct tiled_word_of {
    typedef tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE> type;
};
template<int IN_PER_CYCLE, int OUT_PER_CYCLE, int SHIFT>
struct tiled_word_of<true, IN_PER_CYCLE, OUT_PER_CYCLE, SHIFT> {
    typedef tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, SHIFT> type;
};

// A tap of a step's word. Each mode has its own word, so both branches of the step
// compile for either; the other mode's tap is never used.
template<int IN_PER_CYCLE, int OUT_PER_CYCLE>
weight_t plain_tap(const tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE>& w, int j, int i, int ky, int kx) {
    #pragma HLS INLINE
    return w.v[j][i][ky][kx];
}
template<int IN_PER_CYCLE, int OUT_PER_CYCLE, int SHIFT>
weight_t plain_tap(const tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, SHIFT>&, int, int, int, int) {
    #pragma HLS INLINE
    return 0;
}
template<int SHIFT, int IN_PER_CYCLE, int OUT_PER_CYCLE>
packed_weight_t<SHIFT> packed_tap(const tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, SHIFT>& w,
                                  int pair, int i, int ky, int kx) {
    #pragma HLS INLINE
    return w.v[pair][i][ky][kx];
}
template<int SHIFT, int IN_PER_CYCLE, int OUT_PER_CYCLE>
packed_weight_t<SHIFT> packed_tap(const tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE>&, int, int, int, int) {
    #pragma HLS INLINE
    return 0;
}

// ... and its setters, for the load.
template<int IN_PER_CYCLE, int OUT_PER_CYCLE>
void set_plain_tap(tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE>& w, int j, int i, int ky, int kx, weight_t value) {
    #pragma HLS INLINE
    w.v[j][i][ky][kx] = value;
}
template<int IN_PER_CYCLE, int OUT_PER_CYCLE, int SHIFT>
void set_plain_tap(tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, SHIFT>&, int, int, int, int, weight_t) {}
template<int SHIFT, int IN_PER_CYCLE, int OUT_PER_CYCLE>
void set_packed_tap(tiled_packed_word_t<IN_PER_CYCLE, OUT_PER_CYCLE, SHIFT>& w, int pair, int i, int ky, int kx,
                    packed_weight_t<SHIFT> value) {
    #pragma HLS INLINE
    w.v[pair][i][ky][kx] = value;
}
template<int SHIFT, int IN_PER_CYCLE, int OUT_PER_CYCLE>
void set_packed_tap(tiled_word_t<IN_PER_CYCLE, OUT_PER_CYCLE>&, int, int, int, int, packed_weight_t<SHIFT>) {}

// Output channel j of the step's group done with this step's sum: added to the bias
// or the earlier input groups' sums, and rounded as its output group finishes
// (OUT_PER_CYCLE rounding units, not OUT_CHANNELS).
template<int OUT_GROUPS, int IN_GROUPS, int OUT_PER_CYCLE, int OUT_CHANNELS, activation A, int ROUND_BITS>
void tiled_finish(int j, sum_t sum, int og, int ig, const sum_t bias[OUT_GROUPS][OUT_PER_CYCLE],
                  sum_t running[OUT_PER_CYCLE], typename conv_out<A, ROUND_BITS>::type outputs[OUT_CHANNELS]) {
    #pragma HLS INLINE
    const sum_t total = (ig == 0 ? bias[og][j] : running[j]) + sum;
    running[j] = total;
    const typename conv_out<A, ROUND_BITS>::type done = conv_out<A, ROUND_BITS>::of(total);
    for (int g = 0; g < OUT_GROUPS; g++) {
        #pragma HLS UNROLL
        if (ig == IN_GROUPS - 1 && og == g) outputs[g * OUT_PER_CYCLE + j] = done;
    }
}

template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE,
         activation A, int ROUND_BITS, bool DSP, bool PACKED, int PACK_SHIFT, typename IN_T>
void conv3x3_tiled_patch(
    hls::stream<vec_t<IN_T, IN_CHANNELS> >& in,
    const typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::type
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE)],
    const sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE],
    hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;
    constexpr int TAPS = IN_PER_CYCLE * 9, GROUP = pack_group<PACK_SHIFT, 255>();
    static_assert((MAP_WIDTH & (MAP_WIDTH - 1)) == 0, "a power of two: the centre's y and x are shifts");
    static_assert(!PACKED || (std::is_same<IN_T, uact_t>::value && OUT_PER_CYCLE % 2 == 0),
                  "packed: uint8 inputs and whole pairs of output channels");
    typedef typename conv_out<A, ROUND_BITS>::type out_t;
    typedef typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::type word_t;

    IN_T lines[2][MAP_WIDTH][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    IN_T window[3][3][IN_CHANNELS];
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
                    vec_t<IN_T, IN_CHANNELS> value;
                    if (have) value = in.read();
                    shift_window<MAP_WIDTH, IN_CHANNELS>(lines, window, in_col, value, have);
                }

                // The window's centre is pixel p - LAG.
                if (p >= LAG) {
                    const unsigned centre = p - LAG;
                    const int y = centre / MAP_WIDTH, x = centre % MAP_WIDTH;
                    const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
                    const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
                    const word_t word = weight[og * IN_GROUPS + ig];
                    if (PACKED) {
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
                                    sum += packed_mul<PACK_SHIFT>(packed_tap<PACK_SHIFT>(word, pair, i, ky, kx), a);
                                }
                                unpack_add<PACK_SHIFT>(sum, low, high);
                            }
                            tiled_finish<OUT_GROUPS, IN_GROUPS, OUT_PER_CYCLE, OUT_CHANNELS, A, ROUND_BITS>(
                                2 * pair, low, og, ig, bias, running, outputs);
                            tiled_finish<OUT_GROUPS, IN_GROUPS, OUT_PER_CYCLE, OUT_CHANNELS, A, ROUND_BITS>(
                                2 * pair + 1, high, og, ig, bias, running, outputs);
                        }
                    } else {
                        for (int j = 0; j < OUT_PER_CYCLE; j++) {
                            #pragma HLS UNROLL
                            sum_t sum = 0;
                            for (int i = 0; i < IN_PER_CYCLE; i++) {
                                #pragma HLS UNROLL
                                for (int ky = 0; ky < 3; ky++) {
                                    #pragma HLS UNROLL
                                    for (int kx = 0; kx < 3; kx++) {
                                        #pragma HLS UNROLL
                                        const IN_T a = (row_inside[ky] && col_inside[kx])
                                                     ? window[ky][kx][ig * IN_PER_CYCLE + i] : (IN_T)0;
                                        sum += mul<DSP, sum_t>(a, plain_tap(word, j, i, ky, kx));
                                    }
                                }
                            }
                            tiled_finish<OUT_GROUPS, IN_GROUPS, OUT_PER_CYCLE, OUT_CHANNELS, A, ROUND_BITS>(
                                j, sum, og, ig, bias, running, outputs);
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
// ROM once, when the layer starts; packed, channel 2j + 1's above 2j's.
template<bool PACKED, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, int PACK_SHIFT,
         int BIAS_SHIFT, const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS]>
void load_tiled_weights(
    typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::type
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE)],
    sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE]) {
    constexpr int IN_GROUPS = IN_CHANNELS / IN_PER_CYCLE, OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    load_w: for (int og = 0; og < OUT_GROUPS; og++)
        for (int ig = 0; ig < IN_GROUPS; ig++) {
            #pragma HLS PIPELINE
            typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::type word;
            for (int j = 0; j < (PACKED ? OUT_PER_CYCLE / 2 : OUT_PER_CYCLE); j++) {
                #pragma HLS UNROLL
                for (int i = 0; i < IN_PER_CYCLE; i++) {
                    #pragma HLS UNROLL
                    for (int ky = 0; ky < 3; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            const int in_ch = ig * IN_PER_CYCLE + i;
                            if (PACKED) {
                                const int low = PACKED ? og * OUT_PER_CYCLE + 2 * j : 0;   // in range either way
                                set_packed_tap<PACK_SHIFT>(word, j, i, ky, kx, pack_weights<PACK_SHIFT>(
                                    WEIGHT[low + 1][in_ch][ky][kx], WEIGHT[low][in_ch][ky][kx]));
                            } else {
                                set_plain_tap(word, j, i, ky, kx, WEIGHT[og * OUT_PER_CYCLE + j][in_ch][ky][kx]);
                            }
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
         activation A, int ROUND_BITS, bool PACKED, int PACK_SHIFT, typename IN_T>
void conv3x3_tiled_layer(
    typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::type
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE)],
    hls::stream<vec_t<IN_T, IN_CHANNELS> >& in,
    hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out, int height, int width) {
    #pragma HLS INLINE
    sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    load_tiled_weights<PACKED, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT, BIAS_SHIFT,
                       WEIGHT, BIAS>(weight, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        conv3x3_tiled_patch<MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE, OUT_PER_CYCLE,
                            A, ROUND_BITS, DSP, PACKED, PACK_SHIFT>(in, weight, bias, out);
    }
}

// The weight memory is block RAM when deep, LUTs when shallow (block RAM would
// be mostly empty). impl takes only a literal word: one declaration each.
// PACKED, PACK_SHIFT, IN_T: see conv3x3_tiled_patch (IN_T follows from in).
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS], bool DSP,
         activation A = activation::none, int ROUND_BITS = 0, bool PACKED = false, int PACK_SHIFT = 18,
         typename IN_T = act_t>
void conv3x3_tiled(hls::stream<vec_t<IN_T, IN_CHANNELS> >& in,
                   hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out,
                   int height, int width) {
    constexpr int STEPS = (OUT_CHANNELS / OUT_PER_CYCLE) * (IN_CHANNELS / IN_PER_CYCLE);
    typedef typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::type word_t;
    if (STEPS > 64) {
        word_t weight[STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=bram
        conv3x3_tiled_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
                            OUT_PER_CYCLE, BIAS_SHIFT, WEIGHT, BIAS, DSP, A, ROUND_BITS, PACKED, PACK_SHIFT>(
            weight, in, out, height, width);
    } else {
        word_t weight[STEPS];
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=lutram
        conv3x3_tiled_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
                            OUT_PER_CYCLE, BIAS_SHIFT, WEIGHT, BIAS, DSP, A, ROUND_BITS, PACKED, PACK_SHIFT>(
            weight, in, out, height, width);
    }
}
