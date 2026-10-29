#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "conv3x3_packed.hpp"
#include "conv1d_engine.hpp"

// The layers of a per-pixel (1D) network for the small chips: the input arrives
// one spectral position at a time, the layer keeps a window of the last KERNEL
// positions and folds its work over time, OUT_PER_CYCLE output channels times
// IN_PER_CYCLE input channels a cycle. The pooled outputs leave one position at
// a time again.

// MaxPool1d(2) on a stream of positions: LENGTH in, LENGTH / 2 out.
template<int LENGTH, int CHANNELS>
void pool_pairs(hls::stream<vec_t<uact_t, CHANNELS> >& in, hls::stream<vec_t<uact_t, CHANNELS> >& out,
                int height, int width) {
    const uint32_t pixels = pixel_count(height, width);
    vec_t<uact_t, CHANNELS> even;
    #pragma HLS ARRAY_PARTITION variable=even.v type=complete
    bool odd = false;
    position: for (uint32_t t = 0; t < pixels * LENGTH; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=2 max=66607632
        const vec_t<uact_t, CHANNELS> value = in.read();
        if (odd) {
            vec_t<uact_t, CHANNELS> pooled;
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                pooled.v[ch] = even.v[ch] > value.v[ch] ? even.v[ch] : value.v[ch];
            }
            out.write(pooled);
        } else {
            even = value;
        }
        odd = !odd;
    }
}

// MaxPool1d(2) on positions arriving two at a time (conv1d_engine with two
// positions and every channel a step).
template<int STEPS, int CHANNELS>
void pool_adjacent(hls::stream<vec_t<uact_t, 2 * CHANNELS> >& in, hls::stream<vec_t<uact_t, CHANNELS> >& out,
                   int height, int width) {
    const uint32_t pixels = pixel_count(height, width);
    position: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=33303816
        const vec_t<uact_t, 2 * CHANNELS> pair = in.read();
        vec_t<uact_t, CHANNELS> pooled;
        for (int ch = 0; ch < CHANNELS; ch++) {
            #pragma HLS UNROLL
            pooled.v[ch] = pair.v[ch] > pair.v[CHANNELS + ch] ? pair.v[ch] : pair.v[CHANNELS + ch];
        }
        out.write(pooled);
    }
}

// Conv1d without padding, its first OUT_LENGTH outputs, ReLU and requantization
// to uint8, then MaxPool1d(2).
//
// Each output position takes GROUPS x CHUNKS steps, the first taking the next
// input position in. The first KERNEL - 1 positions of a pixel only fill the
// window; the last output position's free steps read the next pixel's first
// positions ahead, so its fill steps are hidden.
//
// PACKED: two output channels per DSP as in conv3x3_packed.hpp. uint8 products
// are at most 32 640: groups of 1 on the 7-series (PACK_SHIFT 16), 4 on the ZCU104 (18).
template<int IN_LENGTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL, int OUT_LENGTH, int OUT_PER_CYCLE,
         int IN_PER_CYCLE, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][KERNEL], const wgt_t (&BIAS)[OUT_CHANNELS],
         bool PACKED, bool DSP, int PACK_SHIFT = 16, int GROUP = 1>
void conv1d_stream(hls::stream<vec_t<uact_t, IN_CHANNELS> >& in, hls::stream<vec_t<uact_t, OUT_CHANNELS> >& out,
                   int height, int width) {
    constexpr int GROUPS = OUT_CHANNELS / OUT_PER_CYCLE, CHUNKS = IN_CHANNELS / IN_PER_CYCLE;
    constexpr int TAPS = IN_PER_CYCLE * KERNEL;
    constexpr int FILL = KERNEL - 1, TRAILING = IN_LENGTH - OUT_LENGTH - KERNEL + 1;
    constexpr int COMPUTE_STEPS = OUT_LENGTH * GROUPS * CHUNKS;
    constexpr int FREE_STEPS = GROUPS * CHUNKS - 1;   // the last position's steps that take nothing in
    constexpr int READ_AHEAD = TRAILING == 0 ? (FREE_STEPS < FILL ? FREE_STEPS : FILL) : 0;
    constexpr int FILL_NEXT = FILL - READ_AHEAD;      // fill steps left, every pixel after the first
    constexpr int STEPS_PER_PIXEL = FILL_NEXT + COMPUTE_STEPS + TRAILING;

    wgt_t weight[GROUPS * CHUNKS][OUT_PER_CYCLE][TAPS];
    packed_weight_t<PACK_SHIFT> packed[GROUPS * CHUNKS][OUT_PER_CYCLE / 2 > 0 ? OUT_PER_CYCLE / 2 : 1][TAPS];
    acc_t bias[GROUPS][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=packed type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    load_w: for (int g = 0; g < GROUPS; g++)
        for (int c = 0; c < CHUNKS; c++)
            for (int j = 0; j < OUT_PER_CYCLE; j++)
                for (int tap = 0; tap < TAPS; tap++) {
                    #pragma HLS PIPELINE II=1
                    const int o = g * OUT_PER_CYCLE + j, i = c * IN_PER_CYCLE + tap / KERNEL, k = tap % KERNEL;
                    weight[g * CHUNKS + c][j][tap] = WEIGHT[o][i][k];
                    if (PACKED && j % 2 == 1)
                        packed[g * CHUNKS + c][j / 2][tap] =
                            (packed_weight_t<PACK_SHIFT>)WEIGHT[o][i][k] * (packed_weight_t<PACK_SHIFT>)(1 << PACK_SHIFT)
                            + (packed_weight_t<PACK_SHIFT>)WEIGHT[o - 1][i][k];
                    if (c == 0 && tap == 0) bias[g][j] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
                }

    const uint32_t pixels = pixel_count(height, width);
    uact_t window[KERNEL][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
    acc_t sums[OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete
    uact_t position_out[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=position_out type=complete
    uact_t even_position[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=even_position type=complete
    uact_t ahead[FILL > 0 ? FILL : 1][IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=ahead type=complete dim=0
    int pixel_step = 0, position = 0, group = 0, chunk = 0;
    bool first_pixel = true;
    uint32_t pixel = 0;
    const uint32_t steps = pixels == 0 ? 0 : pixels * STEPS_PER_PIXEL + READ_AHEAD;
    step: for (uint32_t cycle = 0; cycle < steps; cycle++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=73790000
        const int fill_steps = first_pixel ? FILL : FILL_NEXT;
        const bool filling = pixel_step < fill_steps;
        const bool trailing = pixel_step >= fill_steps + COMPUTE_STEPS;
        const bool computing = !filling && !trailing;
        const int weight_step = group * CHUNKS + chunk;
        const bool take = filling || trailing || (group == 0 && chunk == 0);
        const bool fetch = READ_AHEAD > 0 && computing && position == OUT_LENGTH - 1
                        && weight_step >= 1 && weight_step <= READ_AHEAD && pixel + 1 < pixels;
        if (!first_pixel && pixel_step == 0) {
            // The positions read ahead go where the remaining fill steps shift them into place.
            for (int p = 0; p < READ_AHEAD; p++)
                for (int ch = 0; ch < IN_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    window[1 + FILL_NEXT + p][ch] = ahead[p][ch];
                }
        }
        // One read for take and fetch, which never meet: two reads would make II = 2.
        vec_t<uact_t, IN_CHANNELS> value;
        if (take || fetch) value = in.read();
        if (fetch) {
            for (int ch = 0; ch < IN_CHANNELS; ch++) {
                #pragma HLS UNROLL
                ahead[weight_step - 1][ch] = value.v[ch];
            }
        }
        if (take) {
            if (!trailing) {
                for (int k = 0; k < KERNEL - 1; k++)
                    for (int ch = 0; ch < IN_CHANNELS; ch++) {
                        #pragma HLS UNROLL
                        window[k][ch] = window[k + 1][ch];
                    }
                for (int ch = 0; ch < IN_CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    window[KERNEL - 1][ch] = value.v[ch];
                }
            }
        }
        if (computing) {
            uact_t window_chunk[KERNEL][IN_PER_CYCLE];
            for (int k = 0; k < KERNEL; k++)
                for (int i = 0; i < IN_PER_CYCLE; i++)
                    for (int c = 0; c < CHUNKS; c++) {
                        #pragma HLS UNROLL
                        if (c == chunk) window_chunk[k][i] = window[k][c * IN_PER_CYCLE + i];
                    }
            if (PACKED) {
                for (int pair = 0; pair < OUT_PER_CYCLE / 2; pair++) {
                    #pragma HLS UNROLL
                    acc_t low = 0, high = 0;
                    for (int first = 0; first < TAPS; first += GROUP) {
                        #pragma HLS UNROLL
                        packed_sum_t sum = 0;
                        for (int tap = first; tap < first + GROUP; tap++) {
                            #pragma HLS UNROLL
                            const packed_sum_t prod = packed[weight_step][pair][tap] * window_chunk[tap % KERNEL][tap / KERNEL];
                            #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                            sum += prod;
                        }
                        low += packed_low<PACK_SHIFT>((int64_t)sum);
                        high += packed_high<PACK_SHIFT>((int64_t)sum);
                    }
                    sums[2 * pair] = (chunk == 0 ? bias[group][2 * pair] : sums[2 * pair]) + low;
                    sums[2 * pair + 1] = (chunk == 0 ? bias[group][2 * pair + 1] : sums[2 * pair + 1]) + high;
                }
            } else {
                for (int j = 0; j < OUT_PER_CYCLE; j++) {
                    #pragma HLS UNROLL
                    acc_t part = 0;
                    for (int tap = 0; tap < TAPS; tap++) {
                        #pragma HLS UNROLL
                        part += mul<DSP, acc_t>(window_chunk[tap % KERNEL][tap / KERNEL], weight[weight_step][j][tap]);
                    }
                    sums[j] = (chunk == 0 ? bias[group][j] : sums[j]) + part;
                }
            }
            if (chunk == CHUNKS - 1) {
                // Indexed by group, not a loop with "if (g == group)": Apple clang 15
                // at -O2 (its SLP vectorizer) got that form wrong.
                for (int j = 0; j < OUT_PER_CYCLE; j++) {
                    #pragma HLS UNROLL
                    position_out[group * OUT_PER_CYCLE + j] = requant_relu8<SHIFT, ACT_MAX>(sums[j]);
                }
                if (group == GROUPS - 1) {
                    if (position % 2 == 0) {
                        for (int ch = 0; ch < OUT_CHANNELS; ch++) {
                            #pragma HLS UNROLL
                            even_position[ch] = position_out[ch];
                        }
                    } else {
                        vec_t<uact_t, OUT_CHANNELS> pooled;
                        for (int ch = 0; ch < OUT_CHANNELS; ch++) {
                            #pragma HLS UNROLL
                            pooled.v[ch] = even_position[ch] > position_out[ch] ? even_position[ch] : position_out[ch];
                        }
                        out.write(pooled);
                    }
                }
            }
            if (chunk < CHUNKS - 1) chunk++;
            else { chunk = 0; if (group < GROUPS - 1) group++; else { group = 0; position++; } }
        }
        if (pixel_step == fill_steps + COMPUTE_STEPS + TRAILING - 1) {
            pixel_step = 0; position = 0; group = 0; chunk = 0; first_pixel = false; pixel++;
        } else {
            pixel_step++;
        }
    }
}

// Flatten (channel-major, as torch.flatten) and the dense layer on positions
// arriving one at a time, FEATURES_PER_CYCLE channels a cycle. Out: the scores,
// shifted left by OUT_SHIFT, as dense_scores.
template<int LENGTH, int CHANNELS, int CLASSES, int FEATURES_PER_CYCLE, int BIAS_SHIFT, int OUT_SHIFT,
         const wgt_t (&WEIGHT)[CLASSES][LENGTH * CHANNELS], const wgt_t (&BIAS)[CLASSES], bool DSP>
void dense_stream(hls::stream<vec_t<uact_t, CHANNELS> >& in, hls::stream<vec_t<acc_t, CLASSES> >& out,
                  int height, int width) {
    constexpr int STEPS_PER_POSITION = CHANNELS / FEATURES_PER_CYCLE, STEPS = LENGTH * STEPS_PER_POSITION;
    wgt_t weight[STEPS][CLASSES][FEATURES_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    load_w: for (int s = 0; s < STEPS; s++)
        for (int c = 0; c < CLASSES; c++)
            for (int i = 0; i < FEATURES_PER_CYCLE; i++) {
                #pragma HLS PIPELINE II=1
                const int pos = s / STEPS_PER_POSITION, ch = (s % STEPS_PER_POSITION) * FEATURES_PER_CYCLE + i;
                weight[s][c][i] = WEIGHT[c][ch * LENGTH + pos];
            }

    const uint32_t pixels = pixel_count(height, width);
    uact_t channels[CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=channels type=complete
    acc_t sums[CLASSES];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete
    int s = 0;
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=48 max=31344768
        if (s % STEPS_PER_POSITION == 0) {
            const vec_t<uact_t, CHANNELS> value = in.read();
            for (int ch = 0; ch < CHANNELS; ch++) {
                #pragma HLS UNROLL
                channels[ch] = value.v[ch];
            }
        } else {
            for (int ch = 0; ch < CHANNELS - FEATURES_PER_CYCLE; ch++) {
                #pragma HLS UNROLL
                channels[ch] = channels[ch + FEATURES_PER_CYCLE];
            }
        }
        for (int c = 0; c < CLASSES; c++) {
            #pragma HLS UNROLL
            acc_t part = 0;
            for (int i = 0; i < FEATURES_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                part += mul<DSP, acc_t>(channels[i], weight[s][c][i]);
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
