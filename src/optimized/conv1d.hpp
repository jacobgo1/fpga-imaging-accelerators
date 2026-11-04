#pragma once

#include <stdint.h>
#include "../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include <type_traits>
#include "dsp_pack.hpp"
#include "pixel_io.hpp"

// The layers of a per-pixel (1D) network, one pixel after another. A conv layer
// is three processes, as the network reads:
//
//   windows (map_windows or position_windows) -> conv1d -> pool (pool_to_map or pool1d)
//
// The windows process cuts a pixel's input into the windows the conv needs, so
// the conv never waits to fill one; the conv computes each window; the pool is
// MaxPool1d(2). The conv is the same on every chip, sized by three numbers:
// POSITIONS outputs a step (along the spectrum), OUT_PER_CYCLE output channels and
// IN_PER_CYCLE input channels; a window takes GROUPS x CHUNKS steps, GROUPS =
// OUT_CHANNELS / OUT_PER_CYCLE, CHUNKS = IN_CHANNELS / IN_PER_CYCLE.
//
// The ZCU104 moves a pixel's whole map between layers (map_windows, pool_to_map);
// the small chips one position at a time, which needs far fewer registers
// (position_windows, pool1d). Every array index into the window is a constant
// (shift registers), so there are no multiplexers in front of the multipliers.

// A pixel's whole map in, its windows out: POSITIONS + KERNEL - 1 positions each,
// POSITIONS apart, enough for the first OUT_LENGTH outputs (no padding).
// Positions past IN_LENGTH are zero.
template<int IN_LENGTH, int CHANNELS, int KERNEL, int OUT_LENGTH, int POSITIONS, typename T>
void map_windows(hls::stream<map_t<T, IN_LENGTH, CHANNELS> >& in,
                 hls::stream<map_t<T, POSITIONS + KERNEL - 1, CHANNELS> >& out, int height, int width) {
    constexpr int WINDOWS = (OUT_LENGTH + POSITIONS - 1) / POSITIONS;
    constexpr int HELD = WINDOWS * POSITIONS + KERNEL - 1;
    const uint32_t pixels = pixel_count(height, width);
    T held[HELD][CHANNELS];   // position 0: this window's first
    #pragma HLS ARRAY_PARTITION variable=held type=complete dim=0
    pixel: for (uint32_t px = 0; px < pixels; px++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=653016
        window: for (int w = 0; w < WINDOWS; w++) {
            #pragma HLS PIPELINE II=1
            if (w == 0) {
                const map_t<T, IN_LENGTH, CHANNELS> map = in.read();
                for (int pos = 0; pos < HELD; pos++)
                    for (int ch = 0; ch < CHANNELS; ch++) {
                        #pragma HLS UNROLL
                        held[pos][ch] = pos < IN_LENGTH ? map.v[pos < IN_LENGTH ? pos : 0][ch] : (T)0;
                    }
            } else {
                for (int pos = 0; pos < HELD; pos++)
                    for (int ch = 0; ch < CHANNELS; ch++) {
                        #pragma HLS UNROLL
                        const int next = pos + POSITIONS;
                        held[pos][ch] = next < HELD ? held[next < HELD ? next : 0][ch] : (T)0;
                    }
            }
            map_t<T, POSITIONS + KERNEL - 1, CHANNELS> window;
            for (int pos = 0; pos < POSITIONS + KERNEL - 1; pos++)
                for (int ch = 0; ch < CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    window.v[pos][ch] = held[pos][ch];
                }
            out.write(window);
        }
    }
}

// Positions in one at a time, IN_LENGTH a pixel; out the windows of KERNEL
// positions for the first OUT_LENGTH outputs. The first KERNEL - 1 positions of a
// pixel only fill the window, and they cost no conv steps: the conv is still on
// the pixel before.
template<int IN_LENGTH, int CHANNELS, int KERNEL, int OUT_LENGTH>
void position_windows(hls::stream<vec_t<uact_t, CHANNELS> >& in,
                      hls::stream<map_t<uact_t, KERNEL, CHANNELS> >& out, int height, int width) {
    static_assert(KERNEL - 1 + OUT_LENGTH <= IN_LENGTH, "no padding");
    const uint32_t pixels = pixel_count(height, width);
    map_t<uact_t, KERNEL, CHANNELS> window;
    #pragma HLS ARRAY_PARTITION variable=window.v type=complete dim=0
    pixel: for (uint32_t px = 0; px < pixels; px++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=653016
        position: for (int pos = 0; pos < IN_LENGTH; pos++) {
            #pragma HLS PIPELINE II=1
            const vec_t<uact_t, CHANNELS> value = in.read();
            for (int k = 0; k < KERNEL; k++)
                for (int ch = 0; ch < CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    window.v[k][ch] = k < KERNEL - 1 ? window.v[k + 1 < KERNEL ? k + 1 : 0][ch] : value.v[ch];
                }
            if (pos >= KERNEL - 1 && pos < KERNEL - 1 + OUT_LENGTH) out.write(window);
        }
    }
}

// Conv1d without padding, then ReLU and requantization to uint8: one window in,
// its POSITIONS outputs with all their channels out. A step does POSITIONS x
// OUT_PER_CYCLE outputs with IN_PER_CYCLE input channels each, every tap of the
// kernel at once; a window takes GROUPS x CHUNKS steps, the bias added at its
// first chunk and the output rounded at its last.
// PACKED: two output channels per DSP (dsp_pack.hpp), uint8 inputs only; PACK_SHIFT
// the chip's packing. Otherwise one multiply each, DSP or LUTs.
template<int IN_CHANNELS, int OUT_CHANNELS, int KERNEL, int OUT_LENGTH, int POSITIONS, int OUT_PER_CYCLE,
         int IN_PER_CYCLE, int BIAS_SHIFT, int ROUND_BITS, int ACT_MAX,
         const weight_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][KERNEL], const weight_t (&BIAS)[OUT_CHANNELS],
         bool PACKED, bool DSP, int PACK_SHIFT, typename IN_T>
void conv1d(hls::stream<map_t<IN_T, POSITIONS + KERNEL - 1, IN_CHANNELS> >& in,
            hls::stream<map_t<uact_t, POSITIONS, OUT_CHANNELS> >& out, int height, int width) {
    constexpr int WINDOWS = (OUT_LENGTH + POSITIONS - 1) / POSITIONS;
    constexpr int GROUPS = OUT_CHANNELS / OUT_PER_CYCLE, CHUNKS = IN_CHANNELS / IN_PER_CYCLE;
    constexpr int TAPS = IN_PER_CYCLE * KERNEL;   // a step's taps for one output: tap = input * KERNEL + k
    constexpr int PAIRS = OUT_PER_CYCLE / 2 > 0 ? OUT_PER_CYCLE / 2 : 1;
    constexpr int GROUP = pack_group<PACK_SHIFT, 255>();
    static_assert(GROUPS * OUT_PER_CYCLE == OUT_CHANNELS && CHUNKS * IN_PER_CYCLE == IN_CHANNELS, "whole steps");
    static_assert(!PACKED || std::is_same<IN_T, uact_t>::value, "packed: uint8 inputs");
    static_assert(!PACKED || (OUT_PER_CYCLE % 2 == 0 && TAPS % GROUP == 0), "whole pairs and groups");

    // The weights a step: one wide word, at address group * CHUNKS + chunk.
    weight_t weight[GROUPS * CHUNKS][OUT_PER_CYCLE][TAPS];
    packed_weight_t<PACK_SHIFT> packed[GROUPS * CHUNKS][PAIRS][TAPS];
    sum_t bias[GROUPS][OUT_PER_CYCLE];
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
                        packed[g * CHUNKS + c][j / 2][tap] = pack_weights<PACK_SHIFT>(WEIGHT[o][i][k], WEIGHT[o - 1][i][k]);
                    if (c == 0 && tap == 0) bias[g][j] = (sum_t)BIAS[o] * ((sum_t)1 << BIAS_SHIFT);
                }

    const uint32_t pixels = pixel_count(height, width);
    map_t<IN_T, POSITIONS + KERNEL - 1, IN_CHANNELS> window;
    #pragma HLS ARRAY_PARTITION variable=window.v type=complete dim=0
    sum_t sums[POSITIONS][OUT_PER_CYCLE];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete dim=0
    map_t<uact_t, POSITIONS, OUT_CHANNELS> outputs;
    #pragma HLS ARRAY_PARTITION variable=outputs.v type=complete dim=0
    windows: for (uint32_t w = 0; w < pixels * WINDOWS; w++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=66607632
        groups: for (int g = 0; g < GROUPS; g++) {
            chunks: for (int c = 0; c < CHUNKS; c++) {
                #pragma HLS PIPELINE II=1
                if (g == 0 && c == 0) window = in.read();
                for (int p = 0; p < POSITIONS; p++) {
                    #pragma HLS UNROLL
                    // This chunk's input channels of position p's window.
                    IN_T taps[KERNEL][IN_PER_CYCLE];
                    for (int k = 0; k < KERNEL; k++)
                        for (int i = 0; i < IN_PER_CYCLE; i++)
                            for (int cc = 0; cc < CHUNKS; cc++) {
                                #pragma HLS UNROLL
                                if (cc == c) taps[k][i] = window.v[p + k][cc * IN_PER_CYCLE + i];
                            }
                    sum_t part[OUT_PER_CYCLE];
                    if (PACKED) {
                        for (int pair = 0; pair < OUT_PER_CYCLE / 2; pair++) {
                            #pragma HLS UNROLL
                            sum_t low = 0, high = 0;
                            for (int first = 0; first < TAPS; first += GROUP) {
                                #pragma HLS UNROLL
                                packed_sum_t sum = 0;
                                for (int tap = first; tap < first + GROUP; tap++) {
                                    #pragma HLS UNROLL
                                    sum += packed_mul<PACK_SHIFT>(packed[g * CHUNKS + c][pair][tap],
                                                                  taps[tap % KERNEL][tap / KERNEL]);
                                }
                                unpack_add<PACK_SHIFT>(sum, low, high);
                            }
                            part[2 * pair] = low;
                            part[2 * pair + 1] = high;
                        }
                    } else {
                        for (int j = 0; j < OUT_PER_CYCLE; j++) {
                            #pragma HLS UNROLL
                            part[j] = 0;
                            for (int tap = 0; tap < TAPS; tap++) {
                                #pragma HLS UNROLL
                                part[j] += mul<DSP, sum_t>(taps[tap % KERNEL][tap / KERNEL], weight[g * CHUNKS + c][j][tap]);
                            }
                        }
                    }
                    for (int j = 0; j < OUT_PER_CYCLE; j++) {
                        #pragma HLS UNROLL
                        sums[p][j] = (c == 0 ? bias[g][j] : sums[p][j]) + part[j];
                        // Indexed by g, not a loop with "if (gg == g)": Apple clang 15
                        // at -O2 (its SLP vectorizer) got that form wrong.
                        if (c == CHUNKS - 1)
                            outputs.v[p][g * OUT_PER_CYCLE + j] = requant_relu8<ROUND_BITS, ACT_MAX>(sums[p][j]);
                    }
                }
                if (g == GROUPS - 1 && c == CHUNKS - 1) out.write(outputs);
            }
        }
    }
}

// conv1d's outputs to the pooled map the next layer reads: MaxPool1d(2) over the
// first 2 x POOLED_LENGTH positions. Each window's positions are shifted in at the
// end of a register array; after a pixel's last window the map goes out.
template<int OUT_LENGTH, int POSITIONS, int CHANNELS, int POOLED_LENGTH>
void pool_to_map(hls::stream<map_t<uact_t, POSITIONS, CHANNELS> >& in,
                 hls::stream<map_t<uact_t, POOLED_LENGTH, CHANNELS> >& out, int height, int width) {
    constexpr int WINDOWS = (OUT_LENGTH + POSITIONS - 1) / POSITIONS;
    constexpr int HELD = WINDOWS * POSITIONS;
    static_assert(2 * POOLED_LENGTH <= HELD, "enough positions");
    uact_t conv[HELD][CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=conv type=complete dim=0
    const uint32_t pixels = pixel_count(height, width);
    pixel: for (uint32_t px = 0; px < pixels; px++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=653016
        window: for (int w = 0; w < WINDOWS; w++) {
            #pragma HLS PIPELINE II=1
            const map_t<uact_t, POSITIONS, CHANNELS> value = in.read();
            for (int pos = 0; pos < HELD; pos++)
                for (int ch = 0; ch < CHANNELS; ch++) {
                    #pragma HLS UNROLL
                    conv[pos][ch] = pos < HELD - POSITIONS ? conv[pos + POSITIONS < HELD ? pos + POSITIONS : 0][ch]
                                                           : value.v[pos >= HELD - POSITIONS ? pos - (HELD - POSITIONS) : 0][ch];
                }
            if (w == WINDOWS - 1) {
                map_t<uact_t, POOLED_LENGTH, CHANNELS> pooled;
                for (int pos = 0; pos < POOLED_LENGTH; pos++)
                    for (int ch = 0; ch < CHANNELS; ch++) {
                        #pragma HLS UNROLL
                        const uact_t a = conv[2 * pos][ch], b = conv[2 * pos + 1][ch];
                        pooled.v[pos][ch] = a > b ? a : b;
                    }
                out.write(pooled);
            }
        }
    }
}

// MaxPool1d(2) to a stream of positions, one out a cycle at most: conv1d's
// outputs one position a window (each second window sends a pooled position) or
// two (each window sends one).
template<int OUT_LENGTH, int POSITIONS, int CHANNELS>
void pool1d(hls::stream<map_t<uact_t, POSITIONS, CHANNELS> >& in, hls::stream<vec_t<uact_t, CHANNELS> >& out,
            int height, int width) {
    static_assert(POSITIONS == 1 || POSITIONS == 2, "one or two positions a window");
    static_assert(OUT_LENGTH % 2 == 0, "whole pairs");
    const uint32_t pixels = pixel_count(height, width);
    uact_t even[CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=even type=complete
    window: for (uint32_t w = 0; w < pixels * (OUT_LENGTH / POSITIONS); w++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=66607632
        const map_t<uact_t, POSITIONS, CHANNELS> value = in.read();
        const bool second = POSITIONS == 2 || w % 2 == 1;   // the pair's second position is here
        vec_t<uact_t, CHANNELS> pooled;
        for (int ch = 0; ch < CHANNELS; ch++) {
            #pragma HLS UNROLL
            const uact_t a = POSITIONS == 2 ? value.v[0][ch] : even[ch], b = value.v[POSITIONS - 1][ch];
            pooled.v[ch] = a > b ? a : b;
            if (!second) even[ch] = value.v[0][ch];
        }
        if (second) out.write(pooled);
    }
}

// Flatten (channel-major, as torch.flatten) and the dense layer, FEATURES_PER_CYCLE
// features a cycle. Out: the scores, shifted left by OUT_SHIFT to their fraction bits.
template<int LENGTH, int CHANNELS, int CLASSES, int FEATURES_PER_CYCLE, int BIAS_SHIFT, int OUT_SHIFT,
         const weight_t (&WEIGHT)[CLASSES][LENGTH * CHANNELS], const weight_t (&BIAS)[CLASSES], bool DSP>
void dense_map(hls::stream<map_t<uact_t, LENGTH, CHANNELS> >& in, hls::stream<vec_t<sum_t, CLASSES> >& out,
               int height, int width) {
    constexpr int FEATURES = LENGTH * CHANNELS, STEPS = FEATURES / FEATURES_PER_CYCLE;
    weight_t weight[STEPS][CLASSES][FEATURES_PER_CYCLE];
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
    sum_t sums[CLASSES];
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
            sum_t part = 0;
            for (int i = 0; i < FEATURES_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                part += mul<DSP, sum_t>(flat[i], weight[s][c][i]);
            }
            sums[c] = (s == 0 ? (sum_t)BIAS[c] * ((sum_t)1 << BIAS_SHIFT) : sums[c]) + part;
        }
        if (s == STEPS - 1) {
            vec_t<sum_t, CLASSES> scores;
            for (int c = 0; c < CLASSES; c++) {
                #pragma HLS UNROLL
                scores.v[c] = sums[c] * ((sum_t)1 << OUT_SHIFT);
            }
            out.write(scores);
            s = 0;
        } else {
            s++;
        }
    }
}

// Flatten (channel-major, as torch.flatten) and the dense layer on positions
// arriving one at a time, FEATURES_PER_CYCLE channels a cycle. Out: the scores,
// shifted left by OUT_SHIFT, as dense_map.
template<int LENGTH, int CHANNELS, int CLASSES, int FEATURES_PER_CYCLE, int BIAS_SHIFT, int OUT_SHIFT,
         const weight_t (&WEIGHT)[CLASSES][LENGTH * CHANNELS], const weight_t (&BIAS)[CLASSES], bool DSP>
void dense_positions(hls::stream<vec_t<uact_t, CHANNELS> >& in, hls::stream<vec_t<sum_t, CLASSES> >& out,
                     int height, int width) {
    constexpr int STEPS_PER_POSITION = CHANNELS / FEATURES_PER_CYCLE, STEPS = LENGTH * STEPS_PER_POSITION;
    weight_t weight[STEPS][CLASSES][FEATURES_PER_CYCLE];
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
    sum_t sums[CLASSES];
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
            sum_t part = 0;
            for (int i = 0; i < FEATURES_PER_CYCLE; i++) {
                #pragma HLS UNROLL
                part += mul<DSP, sum_t>(channels[i], weight[s][c][i]);
            }
            sums[c] = (s == 0 ? (sum_t)BIAS[c] * ((sum_t)1 << BIAS_SHIFT) : sums[c]) + part;
        }
        if (s == STEPS - 1) {
            vec_t<sum_t, CLASSES> scores;
            for (int c = 0; c < CLASSES; c++) {
                #pragma HLS UNROLL
                scores.v[c] = sums[c] * ((sum_t)1 << OUT_SHIFT);
            }
            out.write(scores);
            s = 0;
        } else {
            s++;
        }
    }
}
