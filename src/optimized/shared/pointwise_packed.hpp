#pragma once

#include "conv3x3_packed.hpp"
#include "patch_grid.hpp"

// A 1x1 conv on int8 inputs (the z-scored bands), two multiplies per DSP as in
// conv3x3_packed.hpp, then ReLU and SHIFT bits rounded off. A pixel's bands come
// in chunks of IN_PER_CYCLE, the last padded with zeros; its activations leave
// after the last chunk. OUTPUT_PASSES > 1 splits the channel pairs over that many
// passes per chunk, for a chip with few DSPs. PACK_SHIFT and GROUP: the board's packing.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int PIXELS, int IN_CHANNELS, int OUT_CHANNELS, int IN_PER_CYCLE,
         int BIAS_SHIFT, int SHIFT,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][1][1], const wgt_t (&BIAS)[OUT_CHANNELS],
         int PACK_SHIFT, int GROUP, int OUTPUT_PASSES = 1>
void pointwise_packed(hls::stream<vec_t<act8_t, IN_PER_CYCLE> >& in, hls::stream<vec_t<act_t, OUT_CHANNELS> >& out,
                      int height, int width) {
    constexpr int CHUNKS = (IN_CHANNELS + IN_PER_CYCLE - 1) / IN_PER_CYCLE, PADDED_IN = CHUNKS * IN_PER_CYCLE;
    constexpr int PAIRS_PER_PASS = OUT_CHANNELS / 2 / OUTPUT_PASSES;

    // Pair pass * PAIRS_PER_PASS + p at packed[p][chunk * OUTPUT_PASSES + pass][band].
    packed_weight_t<PACK_SHIFT> packed[PAIRS_PER_PASS][CHUNKS * OUTPUT_PASSES][IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=3
    acc_t bias[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete
    load_w: for (int j = 0; j < OUT_CHANNELS / 2; j++)
        for (int in_ch = 0; in_ch < PADDED_IN; in_ch++) {
            #pragma HLS PIPELINE II=1
            const int source = in_ch < IN_CHANNELS ? in_ch : IN_CHANNELS - 1;   // never read past WEIGHT
            const wgt_t high = in_ch < IN_CHANNELS ? WEIGHT[2 * j + 1][source][0][0] : (wgt_t)0;
            const wgt_t low = in_ch < IN_CHANNELS ? WEIGHT[2 * j][source][0][0] : (wgt_t)0;
            packed[j % PAIRS_PER_PASS][(in_ch / IN_PER_CYCLE) * OUTPUT_PASSES + j / PAIRS_PER_PASS][in_ch % IN_PER_CYCLE] =
                (packed_weight_t<PACK_SHIFT>)high * (packed_weight_t<PACK_SHIFT>)(1 << PACK_SHIFT)
                + (packed_weight_t<PACK_SHIFT>)low;
        }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }

    acc_t sums[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete
    act_t activations[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=activations type=complete
    act8_t chunk_bands[IN_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=chunk_bands type=complete
    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PIXELS;
    int chunk = 0, pass = 0;
    constexpr int TRIPS = PIXELS * CHUNKS * OUTPUT_PASSES, MAX_TRIPS = 665 * TRIPS;
    (void)TRIPS; (void)MAX_TRIPS;   // only the pragma uses them
    px: for (int t = 0; t < pixels * CHUNKS * OUTPUT_PASSES; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=TRIPS max=MAX_TRIPS
        if (pass == 0) {
            const vec_t<act8_t, IN_PER_CYCLE> value = in.read();
            for (int b = 0; b < IN_PER_CYCLE; b++) {
                #pragma HLS UNROLL
                chunk_bands[b] = value.v[b];
            }
        }
        for (int p = 0; p < PAIRS_PER_PASS; p++) {
            #pragma HLS UNROLL
            acc_t low = 0, high = 0;
            for (int first = 0; first < IN_PER_CYCLE; first += GROUP) {
                #pragma HLS UNROLL
                packed_sum_t sum = 0;
                for (int b = first; b < first + GROUP; b++) {
                    #pragma HLS UNROLL
                    const packed_sum_t prod = packed[p][chunk * OUTPUT_PASSES + pass][b] * chunk_bands[b];
                    #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                    sum += prod;
                }
                low += packed_low<PACK_SHIFT>((int64_t)sum);
                high += packed_high<PACK_SHIFT>((int64_t)sum);
            }
            // One adder and one rounding a lane whatever the passes; the result is
            // written by the pass whose turn it is.
            const int j = pass * PAIRS_PER_PASS + p;
            const acc_t low_total = (chunk == 0 ? bias[2 * j] : sums[2 * j]) + low;
            const acc_t high_total = (chunk == 0 ? bias[2 * j + 1] : sums[2 * j + 1]) + high;
            const act_t low_activation = requant_relu<SHIFT>(low_total);
            const act_t high_activation = requant_relu<SHIFT>(high_total);
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
            vec_t<act_t, OUT_CHANNELS> pixel;
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
