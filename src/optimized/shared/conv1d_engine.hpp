#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "conv3x3_packed.hpp"
#include "pixel_io.hpp"

// The layers of a per-pixel (1D) network, one pixel after another. A layer is
// two processes: conv1d_engine does POSITIONS_PER_CYCLE x OUT_PER_CYCLE outputs a
// cycle, every tap at once, and pool_assemble pools them into the next layer's
// map. Each layer is sized to take about the same cycles a pixel. Every array
// index is a constant (shift registers), so there are no multiplexers in front
// of the multipliers.

typedef uint8_t uact_t;   // an activation after ReLU, 0 .. 255

// ReLU, round off SHIFT bits, saturate at ACT_MAX.
template<int SHIFT, int ACT_MAX>
uact_t requant_relu8(acc_t sum) {
    #pragma HLS INLINE
    if (sum <= 0) return 0;
    const acc_t rounded = (sum + ((acc_t)1 << (SHIFT - 1))) >> SHIFT;
    return rounded > ACT_MAX ? (uact_t)ACT_MAX : (uact_t)rounded;
}

// Conv1d without padding, the first OUT_LENGTH outputs of each pixel (the pools
// drop the rest), then ReLU and requantization to uint8. Input positions past
// IN_LENGTH count as zero. Step s sends positions (s / GROUPS) * POSITIONS_PER_CYCLE
// + p, channels (s % GROUPS) * OUT_PER_CYCLE + j, at [p * OUT_PER_CYCLE + j].
// PACKED: two channels per DSP as in conv3x3_packed.hpp; uint8 products are at most
// 32 640, so groups of 4 fit 18 bits. Otherwise one multiply each, DSP or LUTs.
template<int IN_LENGTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL, int OUT_LENGTH, int POSITIONS_PER_CYCLE,
         int OUT_PER_CYCLE, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][KERNEL], const wgt_t (&BIAS)[OUT_CHANNELS],
         bool PACKED, bool DSP, typename IN_T>
void conv1d_engine(hls::stream<map_t<IN_T, IN_LENGTH, IN_CHANNELS> >& in,
                   hls::stream<vec_t<uact_t, POSITIONS_PER_CYCLE * OUT_PER_CYCLE> >& out,
                   int height, int width) {
    constexpr int GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int POSITION_STEPS = (OUT_LENGTH + POSITIONS_PER_CYCLE - 1) / POSITIONS_PER_CYCLE;
    constexpr int STEPS = POSITION_STEPS * GROUPS;
    constexpr int WINDOW_LENGTH = POSITION_STEPS * POSITIONS_PER_CYCLE + KERNEL - 1;
    constexpr int TAPS = IN_CHANNELS * KERNEL;
    constexpr int PACK_SHIFT = 18, GROUP = 4;   // DSP48E2

    wgt_t weight[GROUPS][OUT_PER_CYCLE][TAPS];
    packed_weight_t<PACK_SHIFT> packed[GROUPS][OUT_PER_CYCLE / 2 > 0 ? OUT_PER_CYCLE / 2 : 1][TAPS];
    acc_t bias[GROUPS][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    load_w: for (int g = 0; g < GROUPS; g++)
        for (int j = 0; j < OUT_PER_CYCLE; j++)
            for (int tap = 0; tap < TAPS; tap++) {
                #pragma HLS PIPELINE II=1
                const int o = g * OUT_PER_CYCLE + j;
                weight[g][j][tap] = WEIGHT[o][tap / KERNEL][tap % KERNEL];
                if (PACKED && j % 2 == 1)
                    packed[g][j / 2][tap] =
                        (packed_weight_t<PACK_SHIFT>)WEIGHT[o][tap / KERNEL][tap % KERNEL]
                            * (packed_weight_t<PACK_SHIFT>)(1 << PACK_SHIFT)
                        + (packed_weight_t<PACK_SHIFT>)WEIGHT[o - 1][tap / KERNEL][tap % KERNEL];
                if (tap == 0) bias[g][j] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
            }

    const uint32_t pixels = pixel_count(height, width);
    IN_T window[WINDOW_LENGTH][IN_CHANNELS];   // position 0: this step's first output's first tap
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    int position_step = 0, g = 0;
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=16 max=11754288
        if (position_step == 0 && g == 0) {
            const map_t<IN_T, IN_LENGTH, IN_CHANNELS> map = in.read();
            for (int pos = 0; pos < WINDOW_LENGTH; pos++)
                for (int ch = 0; ch < IN_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    window[pos][ch] = pos < IN_LENGTH ? map.v[pos < IN_LENGTH ? pos : 0][ch] : (IN_T)0;
                }
        } else if (g == 0) {
            for (int pos = 0; pos < WINDOW_LENGTH; pos++)
                for (int ch = 0; ch < IN_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    const int next = pos + POSITIONS_PER_CYCLE;
                    window[pos][ch] = next < WINDOW_LENGTH ? window[next < WINDOW_LENGTH ? next : 0][ch] : (IN_T)0;
                }
        }
        vec_t<uact_t, POSITIONS_PER_CYCLE * OUT_PER_CYCLE> outputs;
        for (int p = 0; p < POSITIONS_PER_CYCLE; p++) {
            #pragma HLS UNROLL
            if (PACKED) {
                for (int pair = 0; pair < OUT_PER_CYCLE / 2; pair++) {
                    #pragma HLS UNROLL
                    acc_t low = bias[g][2 * pair], high = bias[g][2 * pair + 1];
                    for (int first = 0; first < TAPS; first += GROUP) {
                        #pragma HLS UNROLL
                        packed_sum_t sum = 0;
                        for (int tap = first; tap < first + GROUP; tap++) {
                            #pragma HLS UNROLL
                            // The uint8 as it is: widened first, it would need a 27 x 48 multiplier.
                            const packed_sum_t prod = packed[g][pair][tap] * window[p + tap % KERNEL][tap / KERNEL];
                            #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                            sum += prod;
                        }
                        low += packed_low<PACK_SHIFT>((int64_t)sum);
                        high += packed_high<PACK_SHIFT>((int64_t)sum);
                    }
                    outputs.v[p * OUT_PER_CYCLE + 2 * pair] = requant_relu8<SHIFT, ACT_MAX>(low);
                    outputs.v[p * OUT_PER_CYCLE + 2 * pair + 1] = requant_relu8<SHIFT, ACT_MAX>(high);
                }
            } else {
                for (int j = 0; j < OUT_PER_CYCLE; j++) {
                    #pragma HLS UNROLL
                    acc_t sum = bias[g][j];
                    for (int tap = 0; tap < TAPS; tap++) {
                        #pragma HLS UNROLL
                        sum += mul<DSP, acc_t>(window[p + tap % KERNEL][tap / KERNEL], weight[g][j][tap]);
                    }
                    outputs.v[p * OUT_PER_CYCLE + j] = requant_relu8<SHIFT, ACT_MAX>(sum);
                }
            }
        }
        out.write(outputs);
        if (g == GROUPS - 1) { g = 0; position_step = position_step == POSITION_STEPS - 1 ? 0 : position_step + 1; }
        else g++;
    }
}

// conv1d_engine's outputs to the pooled map the next layer reads: MaxPool1d(2)
// over the first 2 x POOLED_LENGTH positions. Each step's values are shifted in
// at the end of a register array; at a pixel's last step the map goes out.
template<int OUT_LENGTH, int POSITIONS_PER_CYCLE, int OUT_PER_CYCLE, int OUT_CHANNELS, int POOLED_LENGTH>
void pool_assemble(hls::stream<vec_t<uact_t, POSITIONS_PER_CYCLE * OUT_PER_CYCLE> >& in,
                   hls::stream<map_t<uact_t, POOLED_LENGTH, OUT_CHANNELS> >& out, int height, int width) {
    constexpr int GROUPS = OUT_CHANNELS / OUT_PER_CYCLE;
    constexpr int POSITION_STEPS = (OUT_LENGTH + POSITIONS_PER_CYCLE - 1) / POSITIONS_PER_CYCLE;
    constexpr int STEPS = POSITION_STEPS * GROUPS;
    constexpr int HELD = POSITION_STEPS * POSITIONS_PER_CYCLE;
    uact_t conv[HELD][OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=conv type=complete dim=0
    const uint32_t pixels = pixel_count(height, width);
    int position_step = 0, g = 0;
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=16 max=11754288
        const vec_t<uact_t, POSITIONS_PER_CYCLE * OUT_PER_CYCLE> value = in.read();
        if (g == 0) {
            for (int pos = 0; pos < HELD - POSITIONS_PER_CYCLE; pos++)
                for (int ch = 0; ch < OUT_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    conv[pos][ch] = conv[pos + POSITIONS_PER_CYCLE][ch];
                }
        }
        // Indexed by g, not a loop over the groups with "if (gg == g)": Apple clang 15
        // at -O2 got that form wrong in conv1d_stream.hpp.
        for (int p = 0; p < POSITIONS_PER_CYCLE; p++)
            for (int j = 0; j < OUT_PER_CYCLE; j++) {
                #pragma HLS UNROLL
                conv[HELD - POSITIONS_PER_CYCLE + p][g * OUT_PER_CYCLE + j] = value.v[p * OUT_PER_CYCLE + j];
            }
        if (position_step == POSITION_STEPS - 1 && g == GROUPS - 1) {
            map_t<uact_t, POOLED_LENGTH, OUT_CHANNELS> pooled;
            for (int pos = 0; pos < POOLED_LENGTH; pos++)
                for (int ch = 0; ch < OUT_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    const uact_t a = conv[2 * pos][ch], b = conv[2 * pos + 1][ch];
                    pooled.v[pos][ch] = a > b ? a : b;
                }
            out.write(pooled);
        }
        if (g == GROUPS - 1) { g = 0; position_step = position_step == POSITION_STEPS - 1 ? 0 : position_step + 1; }
        else g++;
    }
}

// Flatten (channel-major, as torch.flatten) and the dense layer, FEATURES_PER_CYCLE
// features a cycle. Out: the scores, shifted left by OUT_SHIFT to their fraction bits.
template<int LENGTH, int CHANNELS, int CLASSES, int FEATURES_PER_CYCLE, int BIAS_SHIFT, int OUT_SHIFT,
         const wgt_t (&WEIGHT)[CLASSES][LENGTH * CHANNELS], const wgt_t (&BIAS)[CLASSES], bool DSP>
void dense_scores(hls::stream<map_t<uact_t, LENGTH, CHANNELS> >& in, hls::stream<vec_t<acc_t, CLASSES> >& out,
                  int height, int width) {
    constexpr int FEATURES = LENGTH * CHANNELS, STEPS = FEATURES / FEATURES_PER_CYCLE;
    wgt_t weight[STEPS][CLASSES][FEATURES_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    load_w: for (int s = 0; s < STEPS; s++)
        for (int c = 0; c < CLASSES; c++)
            for (int i = 0; i < FEATURES_PER_CYCLE; i++) {
                #pragma HLS PIPELINE II=1
                weight[s][c][i] = WEIGHT[c][s * FEATURES_PER_CYCLE + i];
            }

    const uint32_t pixels = pixel_count(height, width);
    uact_t flat[FEATURES];
    #pragma HLS ARRAY_PARTITION variable=flat type=complete
    acc_t sums[CLASSES];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete
    int s = 0;
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=16 max=10448256
        if (s == 0) {
            const map_t<uact_t, LENGTH, CHANNELS> map = in.read();
            for (int ch = 0; ch < CHANNELS; ch++)
                for (int pos = 0; pos < LENGTH; pos++) {
                    #pragma HLS UNROLL
                    flat[ch * LENGTH + pos] = map.v[pos][ch];
                }
        } else {
            for (int f = 0; f < FEATURES - FEATURES_PER_CYCLE; f++) {
                #pragma HLS UNROLL
                flat[f] = flat[f + FEATURES_PER_CYCLE];
            }
        }
        for (int c = 0; c < CLASSES; c++) {
            #pragma HLS UNROLL
            acc_t part = 0;
            for (int i = 0; i < FEATURES_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                part += mul<DSP, acc_t>(flat[i], weight[s][c][i]);
            }
            sums[c] = (s == 0 ? (acc_t)BIAS[c] * ((acc_t)1 << BIAS_SHIFT) : sums[c]) + part;
        }
        if (s == STEPS - 1) {
            vec_t<acc_t, CLASSES> scores;
            for (int c = 0; c < CLASSES; c++) {
                #pragma HLS UNROLL
                scores.v[c] = sums[c] * ((acc_t)1 << OUT_SHIFT);
            }
            out.write(scores);
            s = 0;
        } else {
            s++;
        }
    }
}
