#pragma once

#include "fixed_point.hpp"
#include "dsp_pack.hpp"
#include "window3x3.hpp"
#include "patch_grid.hpp"

// A 3x3 conv on int8 activations, two multiplies per DSP (dsp_pack.hpp): output
// channels 2j and 2j + 1 share each activation. SHIFT: the chip's packing.
//
// IN_PER_CYCLE input channels a cycle, chunk by chunk: a pixel's sums are done
// in the last chunk, the earlier chunks' wait in partial.

// Packs channels 2j and 2j + 1 into packed[j]; input channels past IN_CHANNELS get
// zero weights, up to PADDED_IN.
template<int OUT_CHANNELS, int IN_CHANNELS, int PADDED_IN, int SHIFT, int BIAS_SHIFT>
void load_packed_weights(const weight_t weight[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t bias[OUT_CHANNELS],
                         packed_weight_t<SHIFT> packed[OUT_CHANNELS / 2][PADDED_IN][3][3],
                         sum_t local_bias[OUT_CHANNELS]) {
    load_w: for (int j = 0; j < OUT_CHANNELS / 2; j++)
        for (int in = 0; in < PADDED_IN; in++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    #pragma HLS PIPELINE II=1
                    const int source = in < IN_CHANNELS ? in : IN_CHANNELS - 1;   // never read past the array
                    const weight_t high = in < IN_CHANNELS ? weight[2 * j + 1][source][ky][kx] : (weight_t)0;
                    const weight_t low = in < IN_CHANNELS ? weight[2 * j][source][ky][kx] : (weight_t)0;
                    packed[j][in][ky][kx] = pack_weights<SHIFT>(high, low);
                }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        local_bias[o] = (sum_t)bias[o] * ((sum_t)1 << BIAS_SHIFT);
    }
}

template<int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE,
         activation A, int ROUND_BITS, int SHIFT>
void conv3x3_2mul_patch(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in,
                        const packed_weight_t<SHIFT> packed[OUT_CHANNELS / 2][IN_CHANNELS][3][3],
                        const sum_t bias[OUT_CHANNELS],
                        hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out) {
    typedef conv_out<A, ROUND_BITS> output;
    constexpr int CHUNKS = IN_CHANNELS / IN_PER_CYCLE;
    constexpr int INPUTS = CHUNKS * MAP_HEIGHT * MAP_WIDTH, LAG = MAP_WIDTH + 1;
    constexpr int GROUP = pack_group<SHIFT, 127>();
    static_assert(IN_PER_CYCLE % GROUP == 0, "whole groups a cycle");

    sum_t partial[CHUNKS > 1 ? MAP_HEIGHT : 1][CHUNKS > 1 ? MAP_WIDTH : 1][OUT_CHANNELS];
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
        shift_window<MAP_WIDTH, IN_PER_CYCLE>(lines, window, in_col, pixel, have);

        if (inputs >= LAG) {
            const bool row_inside[3] = {y > 0, true, y < MAP_HEIGHT - 1};
            const bool col_inside[3] = {x > 0, true, x < MAP_WIDTH - 1};
            vec_t<typename output::type, OUT_CHANNELS> pixel_out;
            for (int j = 0; j < OUT_CHANNELS / 2; j++) {
                #pragma HLS UNROLL
                sum_t low = 0, high = 0;
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
                                sum += packed_mul<SHIFT>(packed[j][chunk * IN_PER_CYCLE + ch][ky][kx], a);
                            }
                            unpack_add<SHIFT>(sum, low, high);
                        }
                    }
                }
                const sum_t low_total = (first_chunk ? bias[2 * j] : partial[y][x][2 * j]) + low;
                const sum_t high_total = (first_chunk ? bias[2 * j + 1] : partial[y][x][2 * j + 1]) + high;
                if (last_chunk) {
                    pixel_out.v[2 * j] = output::of(low_total);
                    pixel_out.v[2 * j + 1] = output::of(high_total);
                } else {
                    partial[y][x][2 * j] = low_total;
                    partial[y][x][2 * j + 1] = high_total;
                }
            }
            if (last_chunk) out.write(pixel_out);
            if (x < MAP_WIDTH - 1) x++;
            else if (y < MAP_HEIGHT - 1) { x = 0; y++; }
            else { x = 0; y = 0; chunk_now++; }
        }
        inputs++;
    }
}

// The layer: the input channels padded with zero weights to a multiple of
// IN_PER_CYCLE, which the producer sends. A: its activation (conv_out), ROUND_BITS
// what that rounds off. The weights stay a ROM inside this process.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int MAP_HEIGHT, int MAP_WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int IN_PER_CYCLE, int BIAS_SHIFT,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][3][3], const weight_t (&BIAS)[OUT_CHANNELS],
         int SHIFT, activation A = activation::none, int ROUND_BITS = 0>
void conv3x3_2mul(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in,
                  hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out,
                  int height, int width) {
    constexpr int PADDED_IN = (IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE * IN_PER_CYCLE;
    packed_weight_t<SHIFT> packed[OUT_CHANNELS / 2][PADDED_IN][3][3];
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=packed type=cyclic factor=IN_PER_CYCLE dim=2
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=4
    sum_t bias[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete

    load_packed_weights<OUT_CHANNELS, IN_CHANNELS, PADDED_IN, SHIFT, BIAS_SHIFT>(WEIGHT, BIAS, packed, bias);
    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    patches: for (int k = 0; k < patches; k++) {
        conv3x3_2mul_patch<MAP_HEIGHT, MAP_WIDTH, PADDED_IN, OUT_CHANNELS, IN_PER_CYCLE, A, ROUND_BITS, SHIFT>(
            in, packed, bias, out);
    }
}
