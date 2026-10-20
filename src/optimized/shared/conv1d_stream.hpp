#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "conv3x3_packed.hpp"   // pack_w_t, pack_s_t, pack_low, pack_high
#include "conv1d_engine.hpp"    // uact_t, requant_relu8, pixel_count

// The layers of a 1D (per-pixel, spectral) network for the small chips (the
// Zynq-7000s): conv1d_engine.hpp's layers pass a pixel's whole feature map at
// once and compute many positions a cycle, which costs registers and wide FIFOs
// a 7020 does not have. Here a layer's input arrives one spectral position at a
// time (all its channels in one word), the layer keeps only a window of the last
// K positions (a 1D line buffer, as conv3x3_stream.hpp keeps two image lines),
// and it folds its work over time: each cycle OCP of its output channels and
// ICP of its input channels, all K taps; the partial sums of the input-channel
// chunks add up in an accumulator. The pooled outputs leave one position at a
// time again.
//
//   positions ─► conv1d_stream ─► pooled positions ─► conv1d_stream ─► ... ─► dense_stream
//
// The window is a shift register (a position comes in at the end, every index a
// constant); the step's input-channel chunk is picked with a C-way choice in front
// of each multiplier (C = IC / ICP, 2 here), as pool_assemble picks a channel group.

// pool_pairs: MaxPool1d(2) on a stream of positions: L positions a pixel in, L / 2
// out (L even: the network only computes the positions its pools keep).
template<int L, int C>
void pool_pairs(hls::stream<vec_t<uact_t, C> >& in, hls::stream<vec_t<uact_t, C> >& out, int height, int width) {
    static_assert(L % 2 == 0, "positions come in pairs");
    const uint32_t pixels = pixel_count(height, width);
    vec_t<uact_t, C> first;
    #pragma HLS ARRAY_PARTITION variable=first.v type=complete
    bool odd = false;
    position: for (uint32_t t = 0; t < pixels * L; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=2 max=66607632
        const vec_t<uact_t, C> v = in.read();
        if (odd) {
            vec_t<uact_t, C> m;
            for (int c = 0; c < C; c++) {
                #pragma HLS UNROLL
                m.v[c] = first.v[c] > v.v[c] ? first.v[c] : v.v[c];
            }
            out.write(m);
        } else {
            first = v;
        }
        odd = !odd;
    }
}

// pool_adjacent: MaxPool1d(2) on positions arriving two at a time, as conv1d_engine
// with POS = 2 and every channel in one step (OCP = C) sends them: STEPS vectors of
// 2 x C a pixel in, each one pooled position out.
template<int STEPS, int C>
void pool_adjacent(hls::stream<vec_t<uact_t, 2 * C> >& in, hls::stream<vec_t<uact_t, C> >& out,
                   int height, int width) {
    const uint32_t pixels = pixel_count(height, width);
    position: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=33303816
        const vec_t<uact_t, 2 * C> v = in.read();
        vec_t<uact_t, C> m;
        for (int c = 0; c < C; c++) {
            #pragma HLS UNROLL
            m.v[c] = v.v[c] > v.v[C + c] ? v.v[c] : v.v[C + c];
        }
        out.write(m);
    }
}

// conv1d_stream: a valid 1D convolution (PyTorch's Conv1d(IC, OC, K)), its first
// L_OUT outputs (the ones the pool after it keeps), ReLU and requantization to
// uint8 (out = clip(round((bias * 2^BIAS_SHIFT + sum(act * q)) / 2^SHIFT), 0,
// ACT_MAX)), then MaxPool1d(2): L_IN positions a pixel in, L_OUT / 2 out.
//
// G = OC / OCP channel groups, C = IC / ICP channel chunks: each output position
// takes G x C steps, the first of them taking the next input position in. The
// first K - 1 positions of a pixel only fill the window. The last output
// position's other G x C - 1 steps take nothing in, so they read the next pixel's
// first positions (up to K - 1) into a side buffer, and that pixel starts with
// them in its window: its fill steps are hidden. A pixel after the first takes
// WARM_NEXT + L_OUT x G x C cycles, WARM_NEXT = K - 1 - min(K - 1, G x C - 1)
// (0 for conv3 and conv4 here, 4 for conv2); the first one K - 1 + L_OUT x G x C.
// (A layer whose inputs go on past what its last output needs, L_IN > L_OUT + K
// - 1, reads and drops them after it, and fills each window as the first.)
// OCP x ICP x K multipliers.
//
// PACKED: two output channels share each activation, so their two weights are
// packed into one multiply (conv3x3_packed.hpp): OCP / 2 x ICP x K DSPs. A
// product is at most 255 x 128 = 32 640 (uint8 activations), and GROUP of them
// must stay below 2^(SHIFT_P - 1): on the 7-series DSP48E1 (25 x 18 bits) the
// packing shift is 16, so GROUP 1 (every product split on its own); on the
// UltraScale+ DSP48E2, 18 and GROUP 4. Otherwise one multiply per product, in
// DSPs (DSP) or LUTs.
template<int L_IN, int IC, int OC, int K, int L_OUT, int OCP, int ICP, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OC][IC][K], const wgt_t (&BIAS)[OC], bool PACKED, bool DSP,
         int SHIFT_P = 16, int GROUP = 1>
void conv1d_stream(hls::stream<vec_t<uact_t, IC> >& in, hls::stream<vec_t<uact_t, OC> >& out,
                   int height, int width) {
    constexpr int G = OC / OCP, C = IC / ICP, F = ICP * K;
    constexpr int WARM = K - 1, TRAIL = L_IN - L_OUT - K + 1, COMPUTE = L_OUT * G * C;
    // The last output position's steps after its first take nothing in: the next
    // pixel's first PRE positions are read then (not when positions trail).
    constexpr int SLOTS = G * C - 1;
    constexpr int PRE = TRAIL == 0 ? (SLOTS < WARM ? SLOTS : WARM) : 0;
    constexpr int WARM_NEXT = WARM - PRE;                     // fill steps left, every pixel after the first
    constexpr int PER_PIXEL = WARM_NEXT + COMPUTE + TRAIL;    // the first pixel PRE more
    static_assert(OC % OCP == 0 && IC % ICP == 0, "OCP must divide OC, ICP must divide IC");
    static_assert(TRAIL >= 0 && L_OUT % 2 == 0, "L_OUT outputs need L_OUT + K - 1 inputs; outputs pool in pairs");
    static_assert(!PACKED || (OCP % 2 == 0 && F % GROUP == 0), "packed: channels in pairs, taps in whole groups");
    static_assert(GROUP * 255 * 128 < (1 << (SHIFT_P - 1)), "a group's low sum must stay inside SHIFT_P signed bits");
    static_assert(BIAS_SHIFT >= 0, "the bias must not have more fraction bits than the sums");

    // The weights, once, into small memories addressed by the step: [group][chunk]
    // x [channel of the group][tap of the chunk], and for PACKED the pairs.
    wgt_t wl[G * C][OCP][F];
    pack_w_t<SHIFT_P> wp[G * C][OCP / 2 > 0 ? OCP / 2 : 1][F];
    acc_t bl[G][OCP];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=bl type=complete dim=2
    load_w: for (int g = 0; g < G; g++)
        for (int c = 0; c < C; c++)
            for (int j = 0; j < OCP; j++)
                for (int f = 0; f < F; f++) {
                    #pragma HLS PIPELINE II=1
                    const int oc = g * OCP + j, ic = c * ICP + f / K, k = f % K;
                    wl[g * C + c][j][f] = WEIGHT[oc][ic][k];
                    if (PACKED && j % 2 == 1)
                        wp[g * C + c][j / 2][f] = (pack_w_t<SHIFT_P>)WEIGHT[oc][ic][k] * (pack_w_t<SHIFT_P>)(1 << SHIFT_P)
                                                  + (pack_w_t<SHIFT_P>)WEIGHT[oc - 1][ic][k];
                    if (c == 0 && f == 0) bl[g][j] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
                }

    const uint32_t pixels = pixel_count(height, width);
    uact_t x[K][IC];           // the window: the last K positions
    #pragma HLS ARRAY_PARTITION variable=x type=complete dim=0
    acc_t acc[OCP];            // this group's sums over the chunks so far
    #pragma HLS ARRAY_PARTITION variable=acc type=complete
    uact_t ov[OC];             // this position's outputs, all groups
    #pragma HLS ARRAY_PARTITION variable=ov type=complete
    uact_t pend[OC];           // the even position's outputs, waiting for its pair
    #pragma HLS ARRAY_PARTITION variable=pend type=complete
    uact_t pre[WARM > 0 ? WARM : 1][IC];   // the next pixel's first positions, read ahead
    #pragma HLS ARRAY_PARTITION variable=pre type=complete dim=0
    int t = 0, o = 0, g = 0, c = 0;   // step of the pixel; output position, group, chunk
    bool first = true;                // the first pixel fills its whole window
    uint32_t pix = 0;
    const uint32_t steps = pixels == 0 ? 0 : pixels * PER_PIXEL + PRE;
    step: for (uint32_t i = 0; i < steps; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=73790000
        const int w = first ? WARM : WARM_NEXT;                       // this pixel's fill steps
        const bool warm = t < w;
        const bool trail = t >= w + COMPUTE;
        const bool compute = !warm && !trail;
        const int s = g * C + c;
        const bool take = warm || trail || (g == 0 && c == 0);       // an input position comes in
        // The last output position's free steps read the next pixel's first positions.
        const bool fetch = PRE > 0 && compute && o == L_OUT - 1 && s >= 1 && s <= PRE && pix + 1 < pixels;
        if (!first && t == 0) {
            // A new pixel: its read-ahead positions into the window, where the
            // remaining fill steps (and its first take) shift them into place.
            for (int p = 0; p < PRE; p++)
                for (int ch = 0; ch < IC; ch++) {
                    #pragma HLS UNROLL
                    x[1 + WARM_NEXT + p][ch] = pre[p][ch];
                }
        }
        // One read for both (they never meet: a take is step 0 of a position, a
        // fetch steps 1 .. PRE). Two reads of the stream in the loop would each
        // need its port, and HLS, not seeing that they exclude each other,
        // would make the loop II = 2: every layer twice as slow.
        vec_t<uact_t, IC> v;
        if (take || fetch) v = in.read();
        if (fetch) {
            for (int ch = 0; ch < IC; ch++) {
                #pragma HLS UNROLL
                pre[s - 1][ch] = v.v[ch];
            }
        }
        if (take) {
            if (!trail) {
                for (int l = 0; l < K - 1; l++)
                    for (int ch = 0; ch < IC; ch++) {
                        #pragma HLS UNROLL
                        x[l][ch] = x[l + 1][ch];
                    }
                for (int ch = 0; ch < IC; ch++) {
                    #pragma HLS UNROLL
                    x[K - 1][ch] = v.v[ch];
                }
            }
        }
        if (compute) {
            // This step's chunk of the window: input channels c x ICP .. c x ICP + ICP - 1.
            uact_t xc[K][ICP];
            for (int k = 0; k < K; k++)
                for (int i = 0; i < ICP; i++)
                    for (int cc = 0; cc < C; cc++) {
                        #pragma HLS UNROLL
                        if (cc == c) xc[k][i] = x[k][cc * ICP + i];
                    }
            if (PACKED) {
                for (int jp = 0; jp < OCP / 2; jp++) {
                    #pragma HLS UNROLL
                    acc_t lo = 0, hi = 0;
                    for (int f0 = 0; f0 < F; f0 += GROUP) {
                        #pragma HLS UNROLL
                        pack_s_t sum = 0;
                        for (int f = f0; f < f0 + GROUP; f++) {
                            #pragma HLS UNROLL
                            // The packed weights times the uint8 itself: 25 x 8 bits, one DSP.
                            const pack_s_t prod = wp[s][jp][f] * xc[f % K][f / K];
                            #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                            sum += prod;
                        }
                        lo += pack_low<SHIFT_P>((int64_t)sum);
                        hi += pack_high<SHIFT_P>((int64_t)sum);
                    }
                    acc[2 * jp] = (c == 0 ? bl[g][2 * jp] : acc[2 * jp]) + lo;
                    acc[2 * jp + 1] = (c == 0 ? bl[g][2 * jp + 1] : acc[2 * jp + 1]) + hi;
                }
            } else {
                for (int j = 0; j < OCP; j++) {
                    #pragma HLS UNROLL
                    acc_t part = 0;
                    for (int f = 0; f < F; f++) {
                        #pragma HLS UNROLL
                        part += mul<DSP, acc_t>(xc[f % K][f / K], wl[s][j][f]);
                    }
                    acc[j] = (c == 0 ? bl[g][j] : acc[j]) + part;
                }
            }
            if (c == C - 1) {
                // This group's channels are done: requantize them into the position's
                // outputs. (Written with the index g, not as a loop over the groups with
                // "if (gg == g)": Apple clang 15 at -O2 got that form wrong, its SLP
                // vectorizer; HLS builds the same G-way choice from either.)
                for (int j = 0; j < OCP; j++) {
                    #pragma HLS UNROLL
                    ov[g * OCP + j] = requant_relu8<SHIFT, ACT_MAX>(acc[j]);
                }
                if (g == G - 1) {
                    // The position is done: pool it with the one before.
                    if (o % 2 == 0) {
                        for (int ch = 0; ch < OC; ch++) {
                            #pragma HLS UNROLL
                            pend[ch] = ov[ch];
                        }
                    } else {
                        vec_t<uact_t, OC> m;
                        for (int ch = 0; ch < OC; ch++) {
                            #pragma HLS UNROLL
                            m.v[ch] = pend[ch] > ov[ch] ? pend[ch] : ov[ch];
                        }
                        out.write(m);
                    }
                }
            }
            if (c < C - 1) c++;
            else { c = 0; if (g < G - 1) g++; else { g = 0; o++; } }
        }
        if (t == w + COMPUTE + TRAIL - 1) { t = 0; o = 0; g = 0; c = 0; first = false; pix++; }
        else t++;
    }
}

// dense_stream: flatten (channel-major, as torch.flatten on (C, L): feature c * L
// + l) and the dense layer, on positions arriving one at a time: FEAT channels of
// the current position x CLASSES a cycle, so L x C / FEAT cycles a pixel. Out: the
// CLASSES sums times 2^OUT_SHIFT, as dense_scores. DSP: its multipliers in DSPs or LUTs.
template<int L, int C, int CLASSES, int FEAT, int BIAS_SHIFT, int OUT_SHIFT,
         const wgt_t (&WEIGHT)[CLASSES][L * C], const wgt_t (&BIAS)[CLASSES], bool DSP>
void dense_stream(hls::stream<vec_t<uact_t, C> >& in, hls::stream<vec_t<acc_t, CLASSES> >& out,
                  int height, int width) {
    constexpr int PER_POS = C / FEAT, STEPS = L * PER_POS;
    static_assert(C % FEAT == 0, "FEAT must divide the channels");
    static_assert(BIAS_SHIFT >= 0 && OUT_SHIFT >= 0, "the scores keep every fraction bit of the sums");
    wgt_t wl[STEPS][CLASSES][FEAT];                        // per step: its features' weights
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    load_w: for (int s = 0; s < STEPS; s++)
        for (int k = 0; k < CLASSES; k++)
            for (int i = 0; i < FEAT; i++) {
                #pragma HLS PIPELINE II=1
                const int l = s / PER_POS, ch = (s % PER_POS) * FEAT + i;
                wl[s][k][i] = WEIGHT[k][ch * L + l];
            }

    const uint32_t pixels = pixel_count(height, width);
    uact_t pv[C];              // the current position's channels, shifted FEAT a step
    #pragma HLS ARRAY_PARTITION variable=pv type=complete
    acc_t acc[CLASSES];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete
    int s = 0;
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=48 max=31344768
        if (s % PER_POS == 0) {
            const vec_t<uact_t, C> v = in.read();
            for (int ch = 0; ch < C; ch++) {
                #pragma HLS UNROLL
                pv[ch] = v.v[ch];
            }
        } else {
            for (int ch = 0; ch < C - FEAT; ch++) {
                #pragma HLS UNROLL
                pv[ch] = pv[ch + FEAT];
            }
        }
        for (int k = 0; k < CLASSES; k++) {
            #pragma HLS UNROLL
            acc_t part = 0;
            for (int i = 0; i < FEAT; i++) {
                #pragma HLS UNROLL
                part += mul<DSP, acc_t>(pv[i], wl[s][k][i]);
            }
            acc[k] = (s == 0 ? (acc_t)BIAS[k] * ((acc_t)1 << BIAS_SHIFT) : acc[k]) + part;
        }
        if (s == STEPS - 1) {
            vec_t<acc_t, CLASSES> o;
            for (int k = 0; k < CLASSES; k++) {
                #pragma HLS UNROLL
                o.v[k] = acc[k] * ((acc_t)1 << OUT_SHIFT);
            }
            out.write(o);
            s = 0;
        } else {
            s++;
        }
    }
}
