#pragma once

#include <type_traits>
#include "fixed_point.hpp"
#include "dsp_pack.hpp"
#include "window3x3.hpp"
#include "patch_grid.hpp"

// A same-padded 3x3 conv of one map, streamed: pixels in raster order in, their
// outputs out, the same order. Two lines of input are kept on chip, the window in
// registers; its centre trails the input by a row and a pixel, so the loop runs
// W + 1 pixels past the last input, and taps outside the map are zeroed by the
// centre's position.
//
// Sized for any time a pixel: a step is the whole window for IN_PER_CYCLE input
// channels times OUT_PER_CYCLE output channels (IN_PER_CYCLE = IN_CHANNELS for a
// small layer); a pixel takes OUT_GROUPS x IN_GROUPS steps, output group outer.
// Out the sums, or with activation::relu ReLU'd and rounded off ROUND_BITS bits
// to activations, so no ReLU module is needed after it.
//
// The input stream's width says how a pixel comes in: all its channels in one word
// (IN_CHANNELS wide, read at the pixel's first step), or, for a first layer of many
// bands (from pixel_buffer), IN_GROUPS chunks of IN_PER_CYCLE, one a step of the
// first output group, the step that first uses it. Chunked, IN_CHANNELS need not
// divide: the last chunk's channels past it get zero weights.
//
// IN_T: the input activations' type, int16, int8 or uint8, multiplied as it is.
// PACKED: two multiplies per DSP (dsp_pack.hpp), 8-bit inputs only: int8 in
// -127 .. 127 or uint8. Output channels 2j and 2j + 1 of a step share every
// activation, so a pair's IN_PER_CYCLE x 9 products go in groups of pack_group
// (int8 8, uint8 4 on the DSP48E2; 2 and 1 on the DSP48E1), the last one shorter
// when they do not divide; each group's sum is split into the two channels.
// PACK_SHIFT: the chip's packing (the kernel's DSP_PACK_SHIFT).
//
// The weights are one wide memory word a step: few memories a layer, since a
// thousand small ones made sp_unet_large_opt too congested to route. HLS packs a
// word at most 4096 bits wide, so a wider one is kept as a few slices of its input
// channels, a memory each (tiled_slices).

// Input groups a pixel, the last one padded; the largest input a packed group must hold.
constexpr int tiled_groups(int channels, int per_cycle) { return (channels + per_cycle - 1) / per_cycle; }
template<typename IN_T> constexpr int input_max() { return std::is_same<IN_T, uact_t>::value ? 255 : 127; }

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

// The most input channels a slice of BITS_PER_INPUT bits each can hold within HLS's
// 4096-bit word, dividing IN_PER_CYCLE so the slices are alike.
constexpr int tiled_slice_inputs(int in_per_cycle, int bits_per_input) {
    int best = 1;
    for (int d = 1; d <= in_per_cycle; d++)
        if (in_per_cycle % d == 0 && d * bits_per_input <= 4096) best = d;
    return best;
}

// A step's word as SLICES memories: slice s holds input channels
// s * IN_PER_SLICE .. (s + 1) * IN_PER_SLICE - 1. One slice when the word fits.
template<bool PACKED, int IN_PER_CYCLE, int OUT_PER_CYCLE, int SHIFT>
struct tiled_slices {
    static constexpr int ROWS = PACKED ? OUT_PER_CYCLE / 2 : OUT_PER_CYCLE;
    static constexpr int WEIGHT_BITS = PACKED ? SHIFT + 9 : 8;
    static constexpr int IN_PER_SLICE = tiled_slice_inputs(IN_PER_CYCLE, ROWS * 9 * WEIGHT_BITS);
    static constexpr int SLICES = IN_PER_CYCLE / IN_PER_SLICE;
    typedef typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, SHIFT>::type word_t;
    typedef typename tiled_word_of<PACKED, IN_PER_SLICE, OUT_PER_CYCLE, SHIFT>::type slice_t;
};

// A step's word from its slices, and back; every index a constant, so only wires.
template<typename S>
typename S::word_t tiled_gather(const typename S::slice_t slices[S::SLICES]) {
    #pragma HLS INLINE
    typename S::word_t word;
    for (int j = 0; j < S::ROWS; j++)
        for (int i = 0; i < S::SLICES * S::IN_PER_SLICE; i++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    #pragma HLS UNROLL
                    word.v[j][i][ky][kx] = slices[i / S::IN_PER_SLICE].v[j][i % S::IN_PER_SLICE][ky][kx];
                }
    return word;
}
template<typename S>
void tiled_scatter(const typename S::word_t& word, typename S::slice_t slices[S::SLICES]) {
    #pragma HLS INLINE
    for (int s = 0; s < S::SLICES; s++) {
        #pragma HLS UNROLL
        typename S::slice_t slice;
        for (int j = 0; j < S::ROWS; j++)
            for (int i = 0; i < S::IN_PER_SLICE; i++)
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++) {
                        #pragma HLS UNROLL
                        slice.v[j][i][ky][kx] = word.v[j][s * S::IN_PER_SLICE + i][ky][kx];
                    }
        slices[s] = slice;
    }
}

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
         activation A, int ROUND_BITS, bool DSP, bool PACKED, int PACK_SHIFT, typename IN_T, int IN_WIDTH>
void conv3x3_tiled_patch(
    hls::stream<vec_t<IN_T, IN_WIDTH> >& in,
    const typename tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::slice_t
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * tiled_groups(IN_CHANNELS, IN_PER_CYCLE)]
              [tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::SLICES],
    const sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE],
    hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out) {
    constexpr int IN_GROUPS = tiled_groups(IN_CHANNELS, IN_PER_CYCLE), OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int PADDED = IN_GROUPS * IN_PER_CYCLE;
    constexpr int IN_WORDS = PADDED / IN_WIDTH;   // words a pixel: 1 whole, IN_GROUPS chunked
    constexpr int PIXELS = MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;
    constexpr int TAPS = IN_PER_CYCLE * 9, GROUP = pack_group<PACK_SHIFT, input_max<IN_T>()>();
    static_assert((MAP_WIDTH & (MAP_WIDTH - 1)) == 0, "a power of two: the centre's y and x are shifts");
    static_assert(IN_WIDTH == PADDED || IN_WIDTH == IN_PER_CYCLE,
                  "a pixel whole (IN_PER_CYCLE dividing IN_CHANNELS) or in chunks of IN_PER_CYCLE");
    static_assert(!PACKED || ((std::is_same<IN_T, uact_t>::value || std::is_same<IN_T, act8_t>::value)
                              && OUT_PER_CYCLE % 2 == 0),
                  "packed: 8-bit inputs and whole pairs of output channels");
    typedef typename conv_out<A, ROUND_BITS>::type out_t;
    typedef typename tiled_word_of<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::type word_t;

    // The lines: a word's channels a memory each, the word of the pixel in the address
    // (a bank chosen at run time would duplicate the stores).
    IN_T lines[2][IN_WORDS * MAP_WIDTH][IN_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    IN_T window[3][3][PADDED];
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
                // The first output group's first IN_WORDS steps take input pixel p in,
                // a word each (whole: step 0 only); past the last pixel, zeros.
                if (og == 0 && ig < IN_WORDS) {
                    const bool have = p < PIXELS;
                    vec_t<IN_T, IN_WIDTH> value;
                    if (have) value = in.read();
                    const int address = (IN_WORDS == 1 ? 0 : ig) * MAP_WIDTH + in_col;
                    for (int c = 0; c < IN_WIDTH; c++) {
                        #pragma HLS UNROLL
                        const IN_T two_up = lines[0][address][c], one_up = lines[1][address][c];
                        const IN_T now = have ? value.v[c] : (IN_T)0;
                        lines[0][address][c] = one_up;
                        lines[1][address][c] = now;
                        for (int w = 0; w < IN_WORDS; w++) {
                            #pragma HLS UNROLL
                            if (IN_WORDS == 1 || w == ig) {
                                const int ch = w * IN_WIDTH + c;
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
                    if (ig == IN_WORDS - 1) in_col = (in_col == MAP_WIDTH - 1) ? 0 : in_col + 1;
                }

                // The window's centre is pixel p - LAG.
                if (p >= LAG) {
                    const unsigned centre = p - LAG;
                    const int y = centre / MAP_WIDTH, x = centre % MAP_WIDTH;
                    const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
                    const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
                    const word_t word = tiled_gather<tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT> >(
                        weight[og * IN_GROUPS + ig]);
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
// ROM once, when the layer starts; packed, channel 2j + 1's above 2j's; zeros
// past IN_CHANNELS.
template<bool PACKED, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int OUT_PER_CYCLE, int PACK_SHIFT,
         int BIAS_SHIFT, const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS]>
void load_tiled_weights(
    typename tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::slice_t
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * tiled_groups(IN_CHANNELS, IN_PER_CYCLE)]
              [tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::SLICES],
    sum_t bias[OUT_CHANNELS / OUT_PER_CYCLE][OUT_PER_CYCLE]) {
    constexpr int IN_GROUPS = tiled_groups(IN_CHANNELS, IN_PER_CYCLE), OUT_GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
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
                            const bool real = in_ch < IN_CHANNELS;
                            const int source = real ? in_ch : IN_CHANNELS - 1;   // never read past the array
                            if (PACKED) {
                                const int low = PACKED ? og * OUT_PER_CYCLE + 2 * j : 0;   // in range either way
                                set_packed_tap<PACK_SHIFT>(word, j, i, ky, kx, pack_weights<PACK_SHIFT>(
                                    real ? WEIGHT[low + 1][source][ky][kx] : (weight_t)0,
                                    real ? WEIGHT[low][source][ky][kx] : (weight_t)0));
                            } else {
                                set_plain_tap(word, j, i, ky, kx,
                                              real ? WEIGHT[og * OUT_PER_CYCLE + j][source][ky][kx] : (weight_t)0);
                            }
                        }
                    }
                }
            }
            tiled_scatter<tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT> >(word, weight[og * IN_GROUPS + ig]);
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
         activation A, int ROUND_BITS, bool PACKED, int PACK_SHIFT, typename IN_T, int IN_WIDTH>
void conv3x3_tiled_layer(
    typename tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::slice_t
        weight[(OUT_CHANNELS / OUT_PER_CYCLE) * tiled_groups(IN_CHANNELS, IN_PER_CYCLE)]
              [tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT>::SLICES],
    hls::stream<vec_t<IN_T, IN_WIDTH> >& in,
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
// PACKED, PACK_SHIFT, IN_T, IN_WIDTH: see conv3x3_tiled_patch (IN_T and IN_WIDTH,
// whole pixel or chunk, follow from in).
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int OUT_PER_CYCLE, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS], bool DSP,
         activation A = activation::none, int ROUND_BITS = 0, bool PACKED = false, int PACK_SHIFT = 18,
         typename IN_T = act_t, int IN_WIDTH = IN_CHANNELS>
void conv3x3_tiled(hls::stream<vec_t<IN_T, IN_WIDTH> >& in,
                   hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out,
                   int height, int width) {
    constexpr int STEPS = (OUT_CHANNELS / OUT_PER_CYCLE) * tiled_groups(IN_CHANNELS, IN_PER_CYCLE);
    typedef tiled_slices<PACKED, IN_PER_CYCLE, OUT_PER_CYCLE, PACK_SHIFT> slices;
    typedef typename slices::slice_t slice_t;
    if (STEPS > 64) {
        slice_t weight[STEPS][slices::SLICES];
        #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=bram
        conv3x3_tiled_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
                            OUT_PER_CYCLE, BIAS_SHIFT, WEIGHT, BIAS, DSP, A, ROUND_BITS, PACKED, PACK_SHIFT>(
            weight, in, out, height, width);
    } else {
        slice_t weight[STEPS][slices::SLICES];
        #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
        #pragma HLS AGGREGATE variable=weight compact=bit
        #pragma HLS BIND_STORAGE variable=weight type=ram_2p impl=lutram
        conv3x3_tiled_layer<PATCH_HEIGHT, PATCH_WIDTH, MAP_HEIGHT, MAP_WIDTH, IN_CHANNELS, OUT_CHANNELS, IN_PER_CYCLE,
                            OUT_PER_CYCLE, BIAS_SHIFT, WEIGHT, BIAS, DSP, A, ROUND_BITS, PACKED, PACK_SHIFT>(
            weight, in, out, height, width);
    }
}
