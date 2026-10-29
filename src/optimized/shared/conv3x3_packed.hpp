#pragma once

#include "fixed_point.hpp"
#include "patch_grid.hpp"

// A 3x3 conv on int8 activations with two multiplies per DSP (AMD WP486). Output
// channels 2j and 2j + 1 share the activation a, so their weights are packed
//     (w_high * 2^SHIFT + w_low) * a = w_high * a * 2^SHIFT + w_low * a
// and a sum of GROUP such products still holds both sums, as long as the low one
// fits SHIFT signed bits (|a| <= 127, never -128). Exact, bit for bit.
//   ZCU104, DSP48E2 (27 bits): SHIFT 18, groups of 8.
//   7-series, DSP48E1 (25 bits): SHIFT 16, groups of 2.
//
// The weights are packed once, at load: packed next to the multiply, HLS put the
// addition in the DSP's pre-adder and the RTL came out wrong (cosim failed).
//
// IN_PER_CYCLE input channels a cycle, chunk by chunk: a pixel's sums are done
// in the last chunk, the earlier chunks' wait in partial.

#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
template<int SHIFT> using packed_weight_t = ap_int<SHIFT + 9>;
typedef ap_int<48> packed_sum_t;                 // the DSP's accumulator
#else
template<int SHIFT> using packed_weight_t = int64_t;
typedef int64_t packed_sum_t;
#endif

// A group's sum to the low channel's part: its bottom SHIFT bits, sign-extended.
template<int SHIFT>
acc_t packed_low(int64_t sum) {
    #pragma HLS INLINE
    return (acc_t)((uint32_t)(uint64_t)sum << (32 - SHIFT)) >> (32 - SHIFT);
}

// ... and the high channel's: the rest, plus the 1 the low part borrowed when negative.
template<int SHIFT>
acc_t packed_high(int64_t sum) {
    #pragma HLS INLINE
    return (acc_t)(sum >> SHIFT) + (acc_t)((sum >> (SHIFT - 1)) & 1);
}

// Packs channels 2j and 2j + 1 into packed[j]; input channels past IN_CHANNELS get
// zero weights, up to PADDED_IN.
template<int OUT_CHANNELS, int IN_CHANNELS, int PADDED_IN, int SHIFT, int BIAS_SHIFT>
void load_packed_weights(const wgt_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t bias[OUT_CHANNELS],
                         packed_weight_t<SHIFT> packed[OUT_CHANNELS / 2][PADDED_IN][3][3],
                         acc_t local_bias[OUT_CHANNELS]) {
    load_w: for (int j = 0; j < OUT_CHANNELS / 2; j++)
        for (int in = 0; in < PADDED_IN; in++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    #pragma HLS PIPELINE II=1
                    const int source = in < IN_CHANNELS ? in : IN_CHANNELS - 1;   // never read past the array
                    const wgt_t high = in < IN_CHANNELS ? weight[2 * j + 1][source][ky][kx] : (wgt_t)0;
                    const wgt_t low = in < IN_CHANNELS ? weight[2 * j][source][ky][kx] : (wgt_t)0;
                    packed[j][in][ky][kx] = (packed_weight_t<SHIFT>)high * (packed_weight_t<SHIFT>)(1 << SHIFT)
                                          + (packed_weight_t<SHIFT>)low;
                }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        local_bias[o] = (acc_t)bias[o] * ((acc_t)1 << BIAS_SHIFT);
    }
}

template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE, int SHIFT, int GROUP>
void conv3x3_packed(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in,
                    const packed_weight_t<SHIFT> packed[OUT_CHANNELS / 2][IN_CHANNELS][3][3],
                    const acc_t bias[OUT_CHANNELS],
                    hls::stream<vec_t<acc_t, OUT_CHANNELS> >& out) {
    constexpr int CHUNKS = IN_CHANNELS / IN_PER_CYCLE;
    constexpr int INPUTS = CHUNKS * MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;

    acc_t partial[CHUNKS > 1 ? MAP_HEIGHT : 1][CHUNKS > 1 ? MAP_WIDTH : 1][OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=partial type=complete dim=3
    act8_t lines[2][MAP_WIDTH][IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lines type=complete dim=3
    act8_t window[3][3][IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0

    int in_col = 0;
    int chunk_now = 0, y = 0, x = 0;   // the window centre's chunk and position
    int inputs = 0;                    // input steps so far, the last W + 1 without input

    conv: for (int t = 0; t < INPUTS + LAG; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=partial type=inter false
        #pragma HLS DEPENDENCE variable=lines type=inter false
        // Spelled out so that with one chunk they are constants.
        const bool first_chunk = CHUNKS == 1 || chunk_now == 0;
        const bool last_chunk = CHUNKS == 1 || chunk_now == CHUNKS - 1;
        const int chunk = CHUNKS == 1 ? 0 : chunk_now;

        const bool have = inputs < INPUTS;
        vec_t<act8_t, IN_PER_CYCLE> pixel;
        if (have) pixel = in.read();
        for (int ch = 0; ch < IN_PER_CYCLE; ch++) {
            #pragma HLS UNROLL
            const act8_t two_up = lines[0][in_col][ch], one_up = lines[1][in_col][ch];
            const act8_t now = have ? pixel.v[ch] : (act8_t)0;
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

        if (inputs >= LAG) {
            const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
            const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
            vec_t<acc_t, OUT_CHANNELS> pixel_sums;
            for (int j = 0; j < OUT_CHANNELS / 2; j++) {
                #pragma HLS UNROLL
                acc_t low = 0, high = 0;
                for (int first = 0; first < IN_PER_CYCLE; first += GROUP) {
                    #pragma HLS UNROLL
                    for (int ky = 0; ky < 3; ky++) {
                        #pragma HLS UNROLL
                        for (int kx = 0; kx < 3; kx++) {
                            #pragma HLS UNROLL
                            packed_sum_t sum = 0;
                            for (int ch = first; ch < first + GROUP; ch++) {
                                #pragma HLS UNROLL
                                const act8_t a = (row_inside[ky] && col_inside[kx]) ? window[ky][kx][ch] : (act8_t)0;
                                const packed_sum_t prod = packed[j][chunk * IN_PER_CYCLE + ch][ky][kx] * a;
                                #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                                sum += prod;
                            }
                            low += packed_low<SHIFT>((int64_t)sum);
                            high += packed_high<SHIFT>((int64_t)sum);
                        }
                    }
                }
                const acc_t low_total = (first_chunk ? bias[2 * j] : partial[y][x][2 * j]) + low;
                const acc_t high_total = (first_chunk ? bias[2 * j + 1] : partial[y][x][2 * j + 1]) + high;
                if (last_chunk) {
                    pixel_sums.v[2 * j] = low_total;
                    pixel_sums.v[2 * j + 1] = high_total;
                } else {
                    partial[y][x][2 * j] = low_total;
                    partial[y][x][2 * j + 1] = high_total;
                }
            }
            if (last_chunk) out.write(pixel_sums);
            if (x < MAP_WIDTH - 1) x++;
            else if (y < MAP_HEIGHT - 1) { x = 0; y++; }
            else { x = 0; y = 0; chunk_now++; }
        }
        inputs++;
    }
}

// The layer: the input channels padded with zero weights to a multiple of
// IN_PER_CYCLE, which the producer sends. The weights stay a ROM inside this process.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int BIAS_SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const wgt_t (&BIAS)[OUT_CHANNELS],
         int SHIFT, int GROUP>
void conv3x3_packed_layer(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in, hls::stream<vec_t<acc_t, OUT_CHANNELS> >& out,
                          int height, int width) {
    constexpr int PADDED_IN = (IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE * IN_PER_CYCLE;
    packed_weight_t<SHIFT> packed[OUT_CHANNELS / 2][PADDED_IN][3][3];
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=packed type=cyclic factor=IN_PER_CYCLE dim=2
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=4
    acc_t bias[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete

    load_packed_weights<OUT_CHANNELS, IN_CHANNELS, PADDED_IN, SHIFT, BIAS_SHIFT>(WEIGHT, BIAS, packed, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_packed<MAP_HEIGHT, MAP_WIDTH, PADDED_IN, OUT_CHANNELS, IN_PER_CYCLE, SHIFT, GROUP>(
            in, packed, bias, out);
    }
}
