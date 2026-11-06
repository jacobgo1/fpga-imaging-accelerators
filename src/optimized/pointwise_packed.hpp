#pragma once

#include "dsp_pack.hpp"
#include "patch_grid.hpp"

// A 1x1 conv on int8 inputs (the z-scored bands), two multiplies per DSP
// (dsp_pack.hpp), then its activation A (conv_out: relu to int16, relu8 to uint8) with
// ROUND_BITS bits rounded off. A pixel's bands come
// in chunks of IN_PER_CYCLE, the last padded with zeros; its activations leave
// after the last chunk. OUTPUT_PASSES > 1 splits the channel pairs over that many
// passes per chunk, for a chip with few DSPs. PACK_SHIFT: the chip's packing.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE,
         int BIAS_SHIFT, int ROUND_BITS,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][1][1], const weight_t (&BIAS)[OUT_CHANNELS],
         int PACK_SHIFT, int OUTPUT_PASSES = 1, activation A = activation::relu>
void pointwise_packed(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in,
                      hls::stream<vec_t<typename conv_out<A, ROUND_BITS>::type, OUT_CHANNELS> >& out,
                      int height, int width) {
    typedef conv_out<A, ROUND_BITS> output;
    typedef typename output::type out_t;
    constexpr int CHUNKS = (IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE, PADDED_IN = CHUNKS * IN_PER_CYCLE;
    constexpr int PAIRS_PER_PASS = OUT_CHANNELS / 2 / OUTPUT_PASSES;
    constexpr int GROUP = pack_group<PACK_SHIFT, 127>();
    static_assert(IN_PER_CYCLE % GROUP == 0, "whole groups a cycle");

    // Pair pass * PAIRS_PER_PASS + p at packed[p][chunk * OUTPUT_PASSES + pass][band].
    packed_weight_t<PACK_SHIFT> packed[PAIRS_PER_PASS][CHUNKS * OUTPUT_PASSES][IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=3
    sum_t bias[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete
    load_w: for (int j = 0; j < OUT_CHANNELS / 2; j++)
        for (int in_ch = 0; in_ch < PADDED_IN; in_ch++) {
            #pragma HLS PIPELINE II=1
            const int source = in_ch < IN_CHANNELS ? in_ch : IN_CHANNELS - 1;   // never read past WEIGHT
            const weight_t high = in_ch < IN_CHANNELS ? WEIGHT[2 * j + 1][source][0][0] : (weight_t)0;
            const weight_t low = in_ch < IN_CHANNELS ? WEIGHT[2 * j][source][0][0] : (weight_t)0;
            packed[j % PAIRS_PER_PASS][(in_ch / IN_PER_CYCLE) * OUTPUT_PASSES + j / PAIRS_PER_PASS][in_ch % IN_PER_CYCLE] =
                pack_weights<PACK_SHIFT>(high, low);
        }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o] = (sum_t)BIAS[o] * ((sum_t)1 << BIAS_SHIFT);
    }

    sum_t sums[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete
    out_t activations[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=activations type=complete
    act8_t chunk_bands[IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=chunk_bands type=complete
    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PIXELS;
    int chunk = 0, pass = 0;
    px: for (int t = 0; t < pixels * CHUNKS * OUTPUT_PASSES; t++) {
        #pragma HLS PIPELINE II=1
        if (pass == 0) {
            const vec_t<act8_t, IN_PER_CYCLE> value = in.read();
            for (int b = 0; b < IN_PER_CYCLE; b++) {
                #pragma HLS UNROLL
                chunk_bands[b] = value.v[b];
            }
        }
        for (int p = 0; p < PAIRS_PER_PASS; p++) {
            #pragma HLS UNROLL
            sum_t low = 0, high = 0;
            for (int first = 0; first < IN_PER_CYCLE; first += GROUP) {
                #pragma HLS UNROLL
                packed_sum_t sum = 0;
                for (int b = first; b < first + GROUP; b++) {
                    #pragma HLS UNROLL
                    sum += packed_mul<PACK_SHIFT>(packed[p][chunk * OUTPUT_PASSES + pass][b], chunk_bands[b]);
                }
                unpack_add<PACK_SHIFT>(sum, low, high);
            }
            // One adder and one rounding a lane whatever the passes; the result is
            // written by the pass whose turn it is.
            const int j = pass * PAIRS_PER_PASS + p;
            const sum_t low_total = (chunk == 0 ? bias[2 * j] : sums[2 * j]) + low;
            const sum_t high_total = (chunk == 0 ? bias[2 * j + 1] : sums[2 * j + 1]) + high;
            const out_t low_activation = output::of(low_total);
            const out_t high_activation = output::of(high_total);
            for (int k = 0; k < OUTPUT_PASSES; k++) {
                #pragma HLS UNROLL
                if (pass == k) {
                    sums[2 * (k * PAIRS_PER_PASS + p)] = low_total;
                    sums[2 * (k * PAIRS_PER_PASS + p) + 1] = high_total;
                    if (chunk == CHUNKS - 1) {
                        activations[2 * (k * PAIRS_PER_PASS + p)] = low_activation;
                        activations[2 * (k * PAIRS_PER_PASS + p) + 1] = high_activation;
                    }
                }
            }
        }
        if (chunk == CHUNKS - 1 && pass == OUTPUT_PASSES - 1) {
            vec_t<out_t, OUT_CHANNELS> pixel;
            for (int o = 0; o < OUT_CHANNELS; o++) {
                #pragma HLS UNROLL
                pixel.v[o] = activations[o];
            }
            out.write(pixel);
        }
        if (pass < OUTPUT_PASSES - 1) {
            pass++;
        } else {
            pass = 0;
            chunk = chunk < CHUNKS - 1 ? chunk + 1 : 0;
        }
    }
}
