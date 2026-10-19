#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "conv3x3_packed.hpp"   // pack_w_t, pack_s_t, pack_low, pack_high
#include "pixel_io.hpp"         // map_t, pixel_count

// The layers of a 1D (per-pixel, spectral) network, streaming one pixel after
// another. A layer is two processes:
//
//   map (pixel's input) ─► conv1d_engine ─► POS x OCP sums a cycle ─► pool_assemble ─► map (pooled)
//
// conv1d_engine does the multiply-adds: every cycle POS output positions x OCP
// output channels, all IC x K taps of each at once (IC x K x POS x OCP
// multipliers), so a pixel takes STEPS = ceil(L_OUT / POS) x (OC / OCP) cycles.
// The kernel picks POS and OCP per layer so that every layer takes about the
// same number of cycles a pixel: then each one's multipliers are busy all the
// time, and they go where the work is. pool_assemble rebuilds the outputs into
// the pooled map the next layer reads.
//
// Every array index in both is a constant: the input window is a shift register
// that moves POS positions a step, and the outputs are shifted in at the end.
// So the hardware has no multiplexers in front of the multipliers, only wires.

typedef uint8_t uact_t;   // an activation after ReLU: never negative, so uint8 (0 .. 255)

// One sum -> one activation: ReLU, round off SHIFT bits, saturate at ACT_MAX.
// Rounding is monotonic, so max-pooling after it gives what pooling before it would.
template<int SHIFT, int ACT_MAX>
uact_t requant_relu8(acc_t v) {
    #pragma HLS INLINE
    static_assert(SHIFT >= 1, "requant_relu8 drops at least one bit");
    if (v <= 0) return 0;
    const acc_t r = (v + ((acc_t)1 << (SHIFT - 1))) >> SHIFT;
    return r > ACT_MAX ? (uact_t)ACT_MAX : (uact_t)r;
}

// conv1d_engine: a valid 1D convolution (PyTorch's Conv1d(IC, OC, K)), the first
// L_OUT outputs of each pixel (the ones the network uses; the pools after it drop
// the rest), then ReLU and requantization to uint8: out = clip(round((bias *
// 2^BIAS_SHIFT + sum(act * q)) / 2^SHIFT), 0, ACT_MAX).
//
// In: one map per pixel, L_IN positions x IC channels (only the first LPAD are
// read; positions past L_IN count as zero). Out: STEPS vectors per pixel, step s
// holding positions (s / GROUPS) * POS + p and channels (s % GROUPS) * OCP + j at
// [p * OCP + j].
//
// PACKED: two output channels share each activation, so their two weights are
// packed into one multiply (conv3x3_packed.hpp, AMD WP486) and one DSP48E2 does
// both: OCP / 2 x POS x IC x K DSPs. The activations must be uint8 (0 .. 255):
// a product is then at most 255 x 128 = 32 640, and GROUP of them must stay
// below 2^(SHIFT_P - 1) = 2^17: GROUP 4. Otherwise one multiply per product, in
// DSPs (DSP) or LUTs, for any activation type (the first layer's 12-bit input).
template<int L_IN, int IC, int OC, int K, int L_OUT, int POS, int OCP, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OC][IC][K], const wgt_t (&BIAS)[OC], bool PACKED, bool DSP, typename IN_T>
void conv1d_engine(hls::stream<map_t<IN_T, L_IN, IC> >& in, hls::stream<vec_t<uact_t, POS * OCP> >& out,
                   int height, int width) {
    constexpr int GROUPS = OC / OCP;                       // output channel groups, one a step
    constexpr int PSTEPS = (L_OUT + POS - 1) / POS;        // position steps
    constexpr int STEPS = PSTEPS * GROUPS;
    constexpr int LPAD = PSTEPS * POS + K - 1;             // input positions the window reaches
    constexpr int F = IC * K;                              // taps of one output
    constexpr int SHIFT_P = 18, GROUP = 4;                 // packing (DSP48E2: 27 x 18 bits)
    static_assert(OC % OCP == 0, "OCP must divide OC");
    static_assert(!PACKED || (OCP % 2 == 0 && F % GROUP == 0), "packed: channels in pairs, taps in groups of 4");
    static_assert(!PACKED || sizeof(IN_T) == 1, "packed: uint8 activations");
    static_assert(GROUP * 255 * 128 < (1 << (SHIFT_P - 1)), "a group's low sum must stay inside SHIFT_P signed bits");
    static_assert(BIAS_SHIFT >= 0, "the bias must not have more fraction bits than the sums");

    // The weights, once, into registers: [group][channel of the group][tap], and
    // for PACKED the pairs [group][pair][tap] = w[odd] * 2^18 + w[even].
    wgt_t wl[GROUPS][OCP][F];
    pack_w_t<SHIFT_P> wp[GROUPS][OCP / 2 > 0 ? OCP / 2 : 1][F];
    acc_t bl[GROUPS][OCP];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wp type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=bl type=complete dim=2
    load_w: for (int g = 0; g < GROUPS; g++)
        for (int j = 0; j < OCP; j++)
            for (int f = 0; f < F; f++) {
                #pragma HLS PIPELINE II=1
                const int oc = g * OCP + j;
                wl[g][j][f] = WEIGHT[oc][f / K][f % K];
                if (PACKED && j % 2 == 1)
                    wp[g][j / 2][f] = (pack_w_t<SHIFT_P>)WEIGHT[oc][f / K][f % K] * (pack_w_t<SHIFT_P>)(1 << SHIFT_P)
                                      + (pack_w_t<SHIFT_P>)WEIGHT[oc - 1][f / K][f % K];
                if (f == 0) bl[g][j] = (acc_t)BIAS[oc] * ((acc_t)1 << BIAS_SHIFT);
            }

    const uint32_t pixels = pixel_count(height, width);
    IN_T x[LPAD][IC];          // the window: position 0 is the current step's first output's first tap
    #pragma HLS ARRAY_PARTITION variable=x type=complete dim=0
    int ps = 0, g = 0;         // position step, channel group
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=16 max=11754288
        if (ps == 0 && g == 0) {
            const map_t<IN_T, L_IN, IC> m = in.read();
            for (int l = 0; l < LPAD; l++)
                for (int c = 0; c < IC; c++) {
                    #pragma HLS UNROLL
                    x[l][c] = l < L_IN ? m.v[l < L_IN ? l : 0][c] : (IN_T)0;
                }
        } else if (g == 0) {
            for (int l = 0; l < LPAD; l++)
                for (int c = 0; c < IC; c++) {
                    #pragma HLS UNROLL
                    x[l][c] = l + POS < LPAD ? x[l + POS < LPAD ? l + POS : 0][c] : (IN_T)0;
                }
        }
        vec_t<uact_t, POS * OCP> o;
        for (int p = 0; p < POS; p++) {
            #pragma HLS UNROLL
            if (PACKED) {
                for (int jp = 0; jp < OCP / 2; jp++) {
                    #pragma HLS UNROLL
                    acc_t lo = bl[g][2 * jp], hi = bl[g][2 * jp + 1];
                    for (int f0 = 0; f0 < F; f0 += GROUP) {
                        #pragma HLS UNROLL
                        pack_s_t s = 0;
                        for (int f = f0; f < f0 + GROUP; f++) {
                            #pragma HLS UNROLL
                            // 27 x 8 bits: the uint8 itself, not widened first (that would
                            // ask for a 27 x 48 multiplier, several DSPs), as in conv3x3_packed.
                            const pack_s_t prod = wp[g][jp][f] * x[p + f % K][f / K];
                            #pragma HLS BIND_OP variable=prod op=mul impl=dsp
                            s += prod;
                        }
                        lo += pack_low<SHIFT_P>((int64_t)s);
                        hi += pack_high<SHIFT_P>((int64_t)s);
                    }
                    o.v[p * OCP + 2 * jp] = requant_relu8<SHIFT, ACT_MAX>(lo);
                    o.v[p * OCP + 2 * jp + 1] = requant_relu8<SHIFT, ACT_MAX>(hi);
                }
            } else {
                for (int j = 0; j < OCP; j++) {
                    #pragma HLS UNROLL
                    acc_t sum = bl[g][j];
                    for (int f = 0; f < F; f++) {
                        #pragma HLS UNROLL
                        sum += mul<DSP, acc_t>(x[p + f % K][f / K], wl[g][j][f]);
                    }
                    o.v[p * OCP + j] = requant_relu8<SHIFT, ACT_MAX>(sum);
                }
            }
        }
        out.write(o);
        if (g == GROUPS - 1) { g = 0; ps = ps == PSTEPS - 1 ? 0 : ps + 1; }
        else g++;
    }
}

// pool_assemble: conv1d_engine's vectors -> the pooled map the next layer reads:
// MaxPool1d(2) over the first 2 x L_POOL positions (the rest are the engine's
// round-up, never used). The positions arrive in order; each position step's POS
// x OC values are shifted in at the end of a register array (every index a
// constant), and at a pixel's last step the pairs are pooled and the map goes out.
template<int L_OUT, int POS, int OCP, int OC, int L_POOL>
void pool_assemble(hls::stream<vec_t<uact_t, POS * OCP> >& in, hls::stream<map_t<uact_t, L_POOL, OC> >& out,
                   int height, int width) {
    constexpr int GROUPS = OC / OCP;
    constexpr int PSTEPS = (L_OUT + POS - 1) / POS;
    constexpr int STEPS = PSTEPS * GROUPS;
    constexpr int LP = PSTEPS * POS;                       // positions held
    static_assert(2 * L_POOL <= LP, "the pool needs 2 x L_POOL positions");
    uact_t conv[LP][OC];
    #pragma HLS ARRAY_PARTITION variable=conv type=complete dim=0
    const uint32_t pixels = pixel_count(height, width);
    int ps = 0, g = 0;
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=16 max=11754288
        const vec_t<uact_t, POS * OCP> v = in.read();
        if (g == 0) {
            // A new position step: make room for its POS positions at the end.
            for (int l = 0; l < LP - POS; l++)
                for (int c = 0; c < OC; c++) {
                    #pragma HLS UNROLL
                    conv[l][c] = conv[l + POS][c];
                }
        }
        // This group's channels of the newest POS positions. With one group (GROUPS
        // == 1) the channel index is a constant; otherwise a GROUPS-way choice.
        // (Written with the index g, not as a loop over the groups with "if (gg ==
        // g)": Apple clang 15 at -O2 got that form wrong in conv1d_stream.hpp.)
        for (int p = 0; p < POS; p++)
            for (int j = 0; j < OCP; j++) {
                #pragma HLS UNROLL
                conv[LP - POS + p][g * OCP + j] = v.v[p * OCP + j];
            }
        if (ps == PSTEPS - 1 && g == GROUPS - 1) {
            map_t<uact_t, L_POOL, OC> m;
            for (int l = 0; l < L_POOL; l++)
                for (int c = 0; c < OC; c++) {
                    #pragma HLS UNROLL
                    const uact_t a = conv[2 * l][c], b = conv[2 * l + 1][c];
                    m.v[l][c] = a > b ? a : b;
                }
            out.write(m);
        }
        if (g == GROUPS - 1) { g = 0; ps = ps == PSTEPS - 1 ? 0 : ps + 1; }
        else g++;
    }
}

// dense_scores: flatten (channel-major, as torch.flatten on (C, L): feature c * L
// + l) and the dense layer, FEAT features x CLASSES a cycle: F / FEAT cycles a
// pixel. Out: the CLASSES sums, times 2^OUT_SHIFT (bias * 2^BIAS_SHIFT + sum(act
// * q), all left-shifted to the scores' fraction bits). DSP: the FEAT x CLASSES
// multipliers in DSPs or LUTs.
template<int L, int C, int CLASSES, int FEAT, int BIAS_SHIFT, int OUT_SHIFT,
         const wgt_t (&WEIGHT)[CLASSES][L * C], const wgt_t (&BIAS)[CLASSES], bool DSP>
void dense_scores(hls::stream<map_t<uact_t, L, C> >& in, hls::stream<vec_t<acc_t, CLASSES> >& out,
                  int height, int width) {
    constexpr int F = L * C, STEPS = F / FEAT;
    static_assert(F % FEAT == 0, "FEAT must divide the features");
    static_assert(BIAS_SHIFT >= 0 && OUT_SHIFT >= 0, "the scores keep every fraction bit of the sums");
    wgt_t wl[STEPS][CLASSES][FEAT];                        // per step: its features' weights
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    load_w: for (int s = 0; s < STEPS; s++)
        for (int c = 0; c < CLASSES; c++)
            for (int i = 0; i < FEAT; i++) {
                #pragma HLS PIPELINE II=1
                wl[s][c][i] = WEIGHT[c][s * FEAT + i];
            }

    const uint32_t pixels = pixel_count(height, width);
    uact_t flat[F];            // the features, shifted FEAT a step
    #pragma HLS ARRAY_PARTITION variable=flat type=complete
    acc_t acc[CLASSES];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete
    int s = 0;
    step: for (uint32_t t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=16 max=10448256
        if (s == 0) {
            const map_t<uact_t, L, C> m = in.read();
            for (int c = 0; c < C; c++)
                for (int l = 0; l < L; l++) {
                    #pragma HLS UNROLL
                    flat[c * L + l] = m.v[l][c];
                }
        } else {
            for (int f = 0; f < F - FEAT; f++) {
                #pragma HLS UNROLL
                flat[f] = flat[f + FEAT];
            }
        }
        for (int c = 0; c < CLASSES; c++) {
            #pragma HLS UNROLL
            acc_t part = 0;
            for (int i = 0; i < FEAT; i++) {
                #pragma HLS UNROLL
                part += mul<DSP, acc_t>(flat[i], wl[s][c][i]);
            }
            acc[c] = (s == 0 ? (acc_t)BIAS[c] * ((acc_t)1 << BIAS_SHIFT) : acc[c]) + part;
        }
        if (s == STEPS - 1) {
            vec_t<acc_t, CLASSES> o;
            for (int c = 0; c < CLASSES; c++) {
                #pragma HLS UNROLL
                o.v[c] = acc[c] * ((acc_t)1 << OUT_SHIFT);
            }
            out.write(o);
            s = 0;
        } else {
            s++;
        }
    }
}
