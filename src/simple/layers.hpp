#pragma once

// The simple kernels' layers: the golden kernels' plain loops (src/golden/layers.hpp) in
// integers, made fast with pragmas only. A kernel calls them once per patch (or block of
// pixels) inside a DATAFLOW loop, so every layer works on a different patch at the same time.
//
// A pixel's channels travel together as one pixel word (an hls::vector). Between two layers
// they go a pixel at a time, in raster order, through a stream; only a 3x3 conv's input is a
// map ([row][col] of pixel words, a ping-pong buffer), since its taps read the neighbours.
// The 1D networks' pixel word holds the whole spectrum, position by position.
//
// The arithmetic is the optimized kernels', so the answers are the same bit for bit: exact
// int32 sums, the bias shifted up to the sums' fraction bits, then rounded half up and the
// activation (see finish).
//
// No element of a pixel word is written conditionally: HLS chains such writes into one
// select after another across the whole word (it missed 250 MHz by 1.3-1.7 ns that way).
// A value chosen at run time lives in a partitioned array, its own register, and the
// word is put together from them.

#include <stdint.h>
#include "../common/hls_vector_shim.hpp"
#include "../common/hls_stream_shim.hpp"

// The maps between the layers are locals of the DATAFLOW loop body (HLS's canonical
// form, so consecutive patches overlap). On the PC they are static: the large network's
// would overflow its 1 MB stack. Vitis's C simulation does not define __SYNTHESIS__.
#ifdef __SYNTHESIS__
#define SIMPLE_MAP
#else
#define SIMPLE_MAP static
#endif

namespace simple {

template<typename T, int N>
using word = hls::vector<T, N>;

// The pixels between two layers. A few deep, so a short wait in one layer does not
// stop the next.
template<typename T, int N>
using stream = hls::stream<word<T, N>, 4>;

typedef hls::vector<uint16_t, 8> cube_word_t;   // 8 raw bands, 128 bits
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// What a layer does to its rounded sums.
enum activation {
    NONE,             // the scores: as they are
    RELU,             // 0 .. ACT_MAX
    CLAMP16,          // saturated to int16
    LEAKY,            // LeakyReLU(0.01) as 41 * 2^-12, rounded half up
    LEAKY_CLAMP16     // both
};

// sum * 2^-SHIFT rounded half up (a negative SHIFT shifts left, exactly), then ACT.
template<int ACT, int ACT_MAX, int SHIFT>
int32_t finish(int32_t sum) {
    #pragma HLS INLINE
    const int64_t r = SHIFT <= 0 ? (int64_t)sum * ((int64_t)1 << (SHIFT < 0 ? -SHIFT : 0))
                                 : ((int64_t)sum + ((int64_t)1 << (SHIFT > 0 ? SHIFT - 1 : 0))) >> SHIFT;
    const int64_t v = (ACT == LEAKY || ACT == LEAKY_CLAMP16) && r < 0 ? (r * 41 + 2048) >> 12 : r;
    if (ACT == RELU) return v < 0 ? 0 : v > ACT_MAX ? ACT_MAX : (int32_t)v;
    if (ACT == CLAMP16 || ACT == LEAKY_CLAMP16) return v < -32768 ? -32768 : v > 32767 ? 32767 : (int32_t)v;
    return (int32_t)v;
}

// Where a conv reads a pixel: a map, at any tap; a stream (a 1x1 conv only), once a pixel
// on its first step, then held.
template<typename W, int N>
W fetch(const W (*map)[N], int y, int x, bool, const W&) {
    #pragma HLS INLINE
    return map[y][x];
}
template<typename W>
W fetch(hls::stream<W>& from, int, int, bool next, const W& held) {
    #pragma HLS INLINE
    return next ? from.read() : held;
}

// Where a layer puts a pixel: a map or a stream.
template<typename W, int N>
void put(W (*map)[N], int y, int x, const W& v) {
    #pragma HLS INLINE
    map[y][x] = v;
}
template<typename W>
void put(hls::stream<W>& to, int, int, const W& v) {
    #pragma HLS INLINE
    to.write(v);
}

// Conv2d, same padding, K = 1 or 3: OUT channels from IN_A channels of a, then IN_B of b
// (torch.cat([a, b], dim=1); IN_B = 0 for one input, b is then never read). UPSAMPLED:
// the inputs are half the size and read at (y/2, x/2), nearest-neighbour upsampling.
//
// One pipelined loop over pixel, output chunk, tap and input chunk: UI inputs times
// UO outputs a step, H*W*K*K*(IN/UI)*(OUT/UO) steps a patch. The taps are not
// unrolled: a step reads one input pixel, so there is no 3x3 window to build.
template<int H, int W, int IN_A, int IN_B, int OUT, int K, int UI, int UO, int ACT, int ACT_MAX,
         int BIAS_SHIFT, int SHIFT, bool UPSAMPLED,
         const int8_t (&WEIGHT)[OUT][IN_A + IN_B][K][K], const int8_t (&BIAS)[OUT],
         typename TI, typename TO, typename WORD_A, typename WORD_B, typename SRC_A, typename SRC_B, typename SINK>
void conv_body(SRC_A& a, SRC_B& b, SINK& out) {
    #pragma HLS INLINE
    constexpr int IN = IN_A + IN_B, PAD = (K - 1) / 2, TAPS = K * K;
    constexpr int IC = IN / UI, IC_A = IN_A / UI, OC = OUT / UO;
    constexpr int STEPS = OC * TAPS * IC;   // a pixel's steps
    static_assert(IN_A % UI == 0 && IN_B % UI == 0, "an input chunk must come from one input");
    static_assert(OUT % UO == 0, "UO must divide the output channels");

    // The weights in the order the steps use them, UO x UI a step, and the biases at the
    // sums' fraction bits. Static and loaded on the first patch only: reloading every
    // patch would take as long as the conv itself in the big layers.
    static int8_t w[OC * TAPS * IC][UO][UI];
    #pragma HLS ARRAY_PARTITION variable=w type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w type=complete dim=3
    static int32_t bias[OC][UO];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    static bool loaded = false;
    if (!loaded) {
        load: for (int o = 0; o < OUT; o++) {
            for (int i = 0; i < IN; i++)
                for (int t = 0; t < TAPS; t++) {
                    #pragma HLS PIPELINE II=1
                    w[((o / UO) * TAPS + t) * IC + i / UI][o % UO][i % UI] = WEIGHT[o][i][t / K][t % K];
                }
            bias[o / UO][o % UO] = (int32_t)BIAS[o] * ((int32_t)1 << BIAS_SHIFT);
        }
        loaded = true;
    }

    int32_t acc[UO] = {};   // each pixel's first step starts from the bias
    #pragma HLS ARRAY_PARTITION variable=acc type=complete
    TO res[OUT];            // the pixel's outputs, a register each
    #pragma HLS ARRAY_PARTITION variable=res type=complete
    WORD_A pa = WORD_A();   // the pixel read, held while a stream's pixel is worked on
    WORD_B pb = WORD_B();
    int row = 0, col = 0, oc = 0, tap = 0, ic = 0;
    steps: for (int s = 0; s < H * W * STEPS; s++) {
        #pragma HLS PIPELINE II=1
        const int y = row + tap / K - PAD, x = col + tap % K - PAD;
        const bool inside = y >= 0 && y < H && x >= 0 && x < W;
        const int sy = inside ? (UPSAMPLED ? y / 2 : y) : 0, sx = inside ? (UPSAMPLED ? x / 2 : x) : 0;
        const bool next = oc == 0 && tap == 0 && ic == 0;   // a new pixel

        // This step's UI inputs; a border tap reads zeros. Each chunk at a fixed place in
        // the word, the run-time choice which of them: a multiplexer.
        pa = fetch(a, sy, sx, next, pa);
        if (IN_B > 0) pb = fetch(b, sy, sx, next, pb);
        TI v[UI];
        #pragma HLS ARRAY_PARTITION variable=v type=complete
        for (int i = 0; i < UI; i++) {
            #pragma HLS UNROLL
            TI chunk[IC];
            #pragma HLS ARRAY_PARTITION variable=chunk type=complete
            for (int g = 0; g < IC; g++) {
                #pragma HLS UNROLL
                chunk[g] = g < IC_A ? pa[g * UI + i] : pb[(IN_B > 0 ? g - IC_A : 0) * UI + i];
            }
            v[i] = inside ? chunk[ic] : (TI)0;
        }

        const int step = (oc * TAPS + tap) * IC + ic;
        const bool first = tap == 0 && ic == 0, last = tap == TAPS - 1 && ic == IC - 1;
        for (int j = 0; j < UO; j++) {
            #pragma HLS UNROLL
            int32_t sum = first ? bias[oc][j] : acc[j];
            for (int i = 0; i < UI; i++) {
                #pragma HLS UNROLL
                sum += (int32_t)v[i] * (int32_t)w[step][j][i];
            }
            acc[j] = sum;
            const TO done = (TO)finish<ACT, ACT_MAX, SHIFT>(sum);
            for (int g = 0; g < OC; g++) {
                #pragma HLS UNROLL
                if (last && oc == g) res[g * UO + j] = done;
            }
        }
        if (last && oc == OC - 1) {
            word<TO, OUT> result;
            for (int k = 0; k < OUT; k++) {
                #pragma HLS UNROLL
                result[k] = res[k];
            }
            put(out, row, col, result);
        }

        // The next step: input chunk, then tap, then output chunk, then pixel.
        if (ic < IC - 1) { ic++; continue; }
        ic = 0;
        if (tap < TAPS - 1) { tap++; continue; }
        tap = 0;
        if (oc < OC - 1) { oc++; continue; }
        oc = 0;
        if (col < W - 1) { col++; continue; }
        col = 0;
        row++;
    }
}

// The usual conv, one input (passed as b too, never read there): from a map or (1x1)
// a stream, to a map or a stream.
#define SIMPLE_CONV_TEMPLATE                                                                         \
    template<int H, int W, int IN, int OUT, int K, int UI, int UO, int ACT, int ACT_MAX,              \
             int BIAS_SHIFT, int SHIFT, bool UPSAMPLED,                                               \
             const int8_t (&WEIGHT)[OUT][IN][K][K], const int8_t (&BIAS)[OUT], typename TI, typename TO>
#define SIMPLE_CONV_BODY                                                                             \
    conv_body<H, W, IN, 0, OUT, K, UI, UO, ACT, ACT_MAX, BIAS_SHIFT, SHIFT, UPSAMPLED, WEIGHT, BIAS,  \
              TI, TO, word<TI, IN>, word<TI, IN> >(in, in, out)
#define SIMPLE_IN_MAP const word<TI, IN> in[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W]

SIMPLE_CONV_TEMPLATE
void conv(SIMPLE_IN_MAP, word<TO, OUT> out[H][W]) { SIMPLE_CONV_BODY; }
SIMPLE_CONV_TEMPLATE
void conv(SIMPLE_IN_MAP, hls::stream<word<TO, OUT> >& out) { SIMPLE_CONV_BODY; }
SIMPLE_CONV_TEMPLATE
void conv(hls::stream<word<TI, IN> >& in, word<TO, OUT> out[H][W]) {
    static_assert(K == 1 && !UPSAMPLED, "a 3x3 or upsampled conv reads a map");
    SIMPLE_CONV_BODY;
}
SIMPLE_CONV_TEMPLATE
void conv(hls::stream<word<TI, IN> >& in, hls::stream<word<TO, OUT> >& out) {
    static_assert(K == 1 && !UPSAMPLED, "a 3x3 or upsampled conv reads a map");
    SIMPLE_CONV_BODY;
}
#undef SIMPLE_CONV_TEMPLATE
#undef SIMPLE_CONV_BODY
#undef SIMPLE_IN_MAP

// The conv after a concat: a's channels, then b's.
template<int H, int W, int IN_A, int IN_B, int OUT, int K, int UI, int UO, int ACT, int ACT_MAX,
         int BIAS_SHIFT, int SHIFT, bool UPSAMPLED,
         const int8_t (&WEIGHT)[OUT][IN_A + IN_B][K][K], const int8_t (&BIAS)[OUT], typename TI, typename TO>
void conv_concat(const word<TI, IN_A> a[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W],
                 const word<TI, IN_B> b[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W], word<TO, OUT> out[H][W]) {
    conv_body<H, W, IN_A, IN_B, OUT, K, UI, UO, ACT, ACT_MAX, BIAS_SHIFT, SHIFT, UPSAMPLED, WEIGHT, BIAS,
              TI, TO, word<TI, IN_A>, word<TI, IN_B> >(a, b, out);
}

// 2 x 2 max pooling, stride 2, a pixel a cycle in raster order: the maxima of an even row's
// pairs wait in a half row for the odd row's.
template<int H, int W, int C, typename T>
void maxpool2(hls::stream<word<T, C> >& in, word<T, C> out[H / 2][W / 2]) {
    word<T, C> half[W / 2];
    // Written on even rows, read on odd ones: W steps apart.
    #pragma HLS DEPENDENCE variable=half type=inter false
    T pair[C];   // the larger of this pair so far
    #pragma HLS ARRAY_PARTITION variable=pair type=complete
    pool: for (int s = 0; s < H * W; s++) {
        #pragma HLS PIPELINE II=1
        const int row = s / W, col = s % W;
        const word<T, C> v = in.read();
        const word<T, C> above = half[col / 2];
        word<T, C> pairs, largest;
        for (int c = 0; c < C; c++) {
            #pragma HLS UNROLL
            const T m = col % 2 == 0 || v[c] > pair[c] ? v[c] : pair[c];
            pair[c] = m;
            pairs[c] = m;
            largest[c] = m > above[c] ? m : above[c];
        }
        if (col % 2 == 1) {
            if (row % 2 == 0) half[col / 2] = pairs;
            else out[row / 2][col / 2] = largest;
        }
    }
}

// One stream to two: a map for the skip connection, a stream for the next layer.
template<int H, int W, int C, typename T>
void duplicate(hls::stream<word<T, C> >& in, word<T, C> first[H][W], hls::stream<word<T, C> >& second) {
    copy: for (int s = 0; s < H * W; s++) {
        #pragma HLS PIPELINE II=1
        const word<T, C> v = in.read();
        first[s / W][s % W] = v;
        second.write(v);
    }
}

// Conv1d, no padding, on a pixel word: L positions x IN channels in (element l * IN + i),
// LO positions x OUT channels out, N pixels. One pipelined loop over pixel, position chunk,
// output chunk and input chunk: UP positions x UO outputs x UI inputs x all K taps a step,
// (LO/UP)*(OUT/UO)*(IN/UI) steps a pixel.
//
// The weights' two layouts: a conv's [OUT][IN][K]; the dense layer's [OUT][IN*K], its input
// flattened channel by channel (torch's flatten of [IN][K]).
template<int IN, int O, int I, int K>
int8_t weight_tap(const int8_t (&w)[O][I][K], int o, int i, int k) {
    return w[o][i][k];
}
template<int IN, int O, int F>
int8_t weight_tap(const int8_t (&w)[O][F], int o, int i, int k) {
    return w[o][i * (F / IN) + k];
}

template<int N, int L, int IN, int OUT, int K, int LO, int UP, int UI, int UO, int ACT, int ACT_MAX,
         int BIAS_SHIFT, int SHIFT, typename WT, const WT& WEIGHT, const int8_t (&BIAS)[OUT],
         typename TI, typename TO>
void conv1d_body(hls::stream<word<TI, L * IN> >& in, hls::stream<word<TO, LO * OUT> >& out) {
    #pragma HLS INLINE
    constexpr int PC = LO / UP, OC = OUT / UO, IC = IN / UI, SPAN = UP + K - 1;
    static_assert(LO % UP == 0 && OUT % UO == 0 && IN % UI == 0, "UP, UO, UI must divide LO, OUT, IN");
    static_assert(LO + K - 1 <= L, "the outputs need inputs past the end");

    // The weights in the order the steps use them, the biases at the sums' fraction bits;
    // loaded on the first block only.
    static int8_t w[OC * IC][UO][UI][K];
    #pragma HLS ARRAY_PARTITION variable=w type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=w type=complete dim=4
    static int32_t bias[OC][UO];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete dim=2
    static bool loaded = false;
    if (!loaded) {
        load: for (int o = 0; o < OUT; o++) {
            for (int i = 0; i < IN; i++)
                for (int k = 0; k < K; k++) {
                    #pragma HLS PIPELINE II=1
                    w[(o / UO) * IC + i / UI][o % UO][i % UI][k] = weight_tap<IN>(WEIGHT, o, i, k);
                }
            bias[o / UO][o % UO] = (int32_t)BIAS[o] * ((int32_t)1 << BIAS_SHIFT);
        }
        loaded = true;
    }

    word<TI, L * IN> x = word<TI, L * IN>();   // the pixel, held for its steps
    int32_t acc[UP][UO];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete
    TO res[LO * OUT];                          // its outputs, a register each
    #pragma HLS ARRAY_PARTITION variable=res type=complete
    int pc = 0, oc = 0, ic = 0;
    steps: for (int s = 0; s < N * PC * OC * IC; s++) {
        #pragma HLS PIPELINE II=1
        if (pc == 0 && oc == 0 && ic == 0) x = in.read();

        // This step's SPAN positions from pc * UP, UI channels from ic * UI: each chunk at a
        // fixed place in the word, the run-time choice which of them.
        TI v[SPAN][UI];
        #pragma HLS ARRAY_PARTITION variable=v type=complete
        for (int q = 0; q < SPAN; q++) {
            #pragma HLS UNROLL
            for (int i = 0; i < UI; i++) {
                #pragma HLS UNROLL
                TI chunk[PC][IC];
                #pragma HLS ARRAY_PARTITION variable=chunk type=complete
                for (int g = 0; g < PC; g++) {
                    #pragma HLS UNROLL
                    for (int h = 0; h < IC; h++) {
                        #pragma HLS UNROLL
                        chunk[g][h] = x[(g * UP + q) * IN + h * UI + i];
                    }
                }
                v[q][i] = chunk[pc][ic];
            }
        }

        const int step = oc * IC + ic;
        const bool first = ic == 0, last = ic == IC - 1;
        for (int p = 0; p < UP; p++) {
            #pragma HLS UNROLL
            for (int j = 0; j < UO; j++) {
                #pragma HLS UNROLL
                int32_t sum = first ? bias[oc][j] : acc[p][j];
                for (int i = 0; i < UI; i++) {
                    #pragma HLS UNROLL
                    for (int k = 0; k < K; k++) {
                        #pragma HLS UNROLL
                        sum += (int32_t)v[p + k][i] * (int32_t)w[step][j][i][k];
                    }
                }
                acc[p][j] = sum;
                const TO done = (TO)finish<ACT, ACT_MAX, SHIFT>(sum);
                for (int g = 0; g < PC; g++) {
                    #pragma HLS UNROLL
                    for (int h = 0; h < OC; h++) {
                        #pragma HLS UNROLL
                        if (last && pc == g && oc == h) res[(g * UP + p) * OUT + h * UO + j] = done;
                    }
                }
            }
        }
        if (last && oc == OC - 1 && pc == PC - 1) {
            word<TO, LO * OUT> result;
            for (int e = 0; e < LO * OUT; e++) {
                #pragma HLS UNROLL
                result[e] = res[e];
            }
            out.write(result);
        }

        // The next step: input chunk, then output chunk, then position chunk, then pixel.
        if (ic < IC - 1) { ic++; continue; }
        ic = 0;
        if (oc < OC - 1) { oc++; continue; }
        oc = 0;
        pc = pc < PC - 1 ? pc + 1 : 0;
    }
}

// Conv1d (no padding) on N pixel words, L positions x IN channels to L-K+1 x OUT.
template<int N, int L, int IN, int OUT, int K, int UP, int UI, int UO, int ACT, int ACT_MAX, int BIAS_SHIFT, int SHIFT,
         const int8_t (&WEIGHT)[OUT][IN][K], const int8_t (&BIAS)[OUT], typename TI, typename TO>
void conv1d(hls::stream<word<TI, L * IN> >& in, hls::stream<word<TO, (L - K + 1) * OUT> >& out) {
    conv1d_body<N, L, IN, OUT, K, L - K + 1, UP, UI, UO, ACT, ACT_MAX, BIAS_SHIFT, SHIFT,
                int8_t[OUT][IN][K], WEIGHT, BIAS, TI, TO>(in, out);
}

// Flatten (channel by channel) and a dense layer, on N pixel words of L positions x IN
// channels: a conv1d whose K taps are all L positions. The scores, no activation.
template<int N, int L, int IN, int OUT, int UI, int UO, int BIAS_SHIFT, int SHIFT,
         const int8_t (&WEIGHT)[OUT][L * IN], const int8_t (&BIAS)[OUT], typename TI>
void dense(hls::stream<word<TI, L * IN> >& in, hls::stream<word<int32_t, OUT> >& out) {
    conv1d_body<N, L, IN, OUT, L, 1, 1, UI, UO, NONE, 0, BIAS_SHIFT, SHIFT,
                int8_t[OUT][L * IN], WEIGHT, BIAS, TI, int32_t>(in, out);
}

// Max pooling by 2 along a pixel word's L positions of C channels (an odd last one
// dropped), N pixels, one a cycle.
template<int N, int L, int C, typename T>
void maxpool1d(hls::stream<word<T, L * C> >& in, hls::stream<word<T, (L / 2) * C> >& out) {
    pool: for (int n = 0; n < N; n++) {
        #pragma HLS PIPELINE II=1
        const word<T, L * C> x = in.read();
        word<T, (L / 2) * C> y;
        for (int l = 0; l < L / 2; l++) {
            #pragma HLS UNROLL
            for (int c = 0; c < C; c++) {
                #pragma HLS UNROLL
                const T a = x[2 * l * C + c], b = x[(2 * l + 1) * C + c];
                y[l * C + c] = a > b ? a : b;
            }
        }
        out.write(y);
    }
}

// One level of a 1D wavelet transform along each pixel's BANDS bands (golden::dwt1d_concat):
// [x * 2^FRAC, low, high], S = (BANDS + TAPS - 1) / 2 of each, the bands padded with zeros
// at both ends, stride 2. UM low and high outputs a step, S / UM steps a pixel. The filters
// are constants, so their multiplies are shifts and adds. N pixels.
template<int N, int BANDS, int TAPS, int UM, int FRAC, const int8_t (&LOW)[1][1][TAPS],
         const int8_t (&HIGH)[1][1][TAPS], typename TI, typename TO>
void dwt(hls::stream<word<TI, BANDS> >& in, hls::stream<word<TO, BANDS + 2 * ((BANDS + TAPS - 1) / 2)> >& out) {
    constexpr int S = (BANDS + TAPS - 1) / 2, PAD = (2 * (S - 1) - BANDS + TAPS) / 2;
    constexpr int MC = S / UM, SPAN = 2 * UM + TAPS - 2;
    static_assert(S % UM == 0, "UM must divide the outputs");
    word<TI, BANDS> x = word<TI, BANDS>();
    TO low[S], high[S];
    #pragma HLS ARRAY_PARTITION variable=low type=complete
    #pragma HLS ARRAY_PARTITION variable=high type=complete
    int mc = 0;
    steps: for (int s = 0; s < N * MC; s++) {
        #pragma HLS PIPELINE II=1
        if (mc == 0) x = in.read();

        // This step's SPAN bands from 2 * UM * mc - PAD (zero outside), each chunk at a
        // fixed place, the run-time choice which of them.
        int32_t v[SPAN];
        #pragma HLS ARRAY_PARTITION variable=v type=complete
        for (int q = 0; q < SPAN; q++) {
            #pragma HLS UNROLL
            int32_t chunk[MC];
            #pragma HLS ARRAY_PARTITION variable=chunk type=complete
            for (int g = 0; g < MC; g++) {
                #pragma HLS UNROLL
                const int band = 2 * UM * g + q - PAD;
                chunk[g] = band >= 0 && band < BANDS ? (int32_t)x[band >= 0 && band < BANDS ? band : 0] : 0;
            }
            v[q] = chunk[mc];
        }
        for (int u = 0; u < UM; u++) {
            #pragma HLS UNROLL
            int32_t lo = 0, hi = 0;
            for (int k = 0; k < TAPS; k++) {
                #pragma HLS UNROLL
                lo += v[2 * u + k] * LOW[0][0][k];
                hi += v[2 * u + k] * HIGH[0][0][k];
            }
            for (int g = 0; g < MC; g++) {
                #pragma HLS UNROLL
                if (mc == g) {
                    low[g * UM + u] = (TO)lo;
                    high[g * UM + u] = (TO)hi;
                }
            }
        }
        if (mc == MC - 1) {
            word<TO, BANDS + 2 * S> y;
            for (int c = 0; c < BANDS; c++) {
                #pragma HLS UNROLL
                y[c] = (TO)((int32_t)x[c] * (1 << FRAC));
            }
            for (int m = 0; m < S; m++) {
                #pragma HLS UNROLL
                y[BANDS + m] = low[m];
                y[BANDS + S + m] = high[m];
            }
            out.write(y);
        }
        mc = mc < MC - 1 ? mc + 1 : 0;
    }
}

// INSIDE consecutive pixels of the raw cube from DDR in one burst, z-scored into the
// line's pixels 0 .. INSIDE - 1; the rest of the line's N repeat the last of them. Words
// before FIRST_WORD are dropped; kept band k is (dn * A[k] + B[k]) >> FRAC, saturated to
// +-IN_MAX. B_T: B's type, int32 or int64. LINE: a map row (a pointer) or a stream.
template<typename W>
void store(W* line, int c, const W& v) {
    #pragma HLS INLINE
    line[c] = v;
}
template<typename W>
void store(hls::stream<W>& line, int, const W& v) {
    #pragma HLS INLINE
    line.write(v);
}

template<int N, int BANDS, int WORDS_PER_PIXEL, int FIRST_WORD, int FRAC, int IN_MAX, typename B_T,
         const int32_t (&A)[(WORDS_PER_PIXEL - FIRST_WORD) * 8], const B_T (&B)[(WORDS_PER_PIXEL - FIRST_WORD) * 8],
         typename T, typename LINE>
void read_line(const cube_word_t* from, int inside, LINE& line) {
    #pragma HLS INLINE
    constexpr int KEPT_WORDS = WORDS_PER_PIXEL - FIRST_WORD;
    T band_value[BANDS];   // the pixel's z-scores, a register each
    #pragma HLS ARRAY_PARTITION variable=band_value type=complete
    word<T, BANDS> pixel = word<T, BANDS>();
    int w = 0, col = 0;
    take: for (int k = 0; k < inside * WORDS_PER_PIXEL; k++) {
        #pragma HLS PIPELINE II=1
        const cube_word_t raw = from[k];
        for (int b = 0; b < 8; b++) {
            #pragma HLS UNROLL
            // Kept word kw at the fixed place kw: the run-time choice is which runs.
            for (int kw = 0; kw < KEPT_WORDS; kw++) {
                #pragma HLS UNROLL
                const int band = kw * 8 + b;
                if (band < BANDS && w == FIRST_WORD + kw) {
                    const int64_t z = ((int64_t)raw[b] * A[band] + B[band]) >> FRAC;
                    band_value[band < BANDS ? band : 0] = (T)(z > IN_MAX ? IN_MAX : z < -IN_MAX ? -IN_MAX : z);
                }
            }
        }
        if (w == WORDS_PER_PIXEL - 1) {
            for (int c = 0; c < BANDS; c++) {
                #pragma HLS UNROLL
                pixel[c] = band_value[c];
            }
            store(line, col, pixel);
            col++;
            w = 0;
        } else {
            w++;
        }
    }
    edge: for (int c = inside; c < N; c++) {
        #pragma HLS PIPELINE II=1
        store(line, c, pixel);
    }
}

// Each of INSIDE pixels' scores, then its class (the first highest score), to DDR in one burst.
template<int N, int CLASSES>
void write_line(const word<int32_t, CLASSES> line[N], score_word_t* to, int inside) {
    #pragma HLS INLINE
    put: for (int c = 0; c < inside; c++) {
        #pragma HLS PIPELINE II=1
        const word<int32_t, CLASSES> s = line[c];
        score_word_t out;
        int best = 0;
        for (int k = 0; k < CLASSES; k++) {
            #pragma HLS UNROLL
            out[k] = s[k];
        }
        for (int k = 1; k < CLASSES; k++) {
            #pragma HLS UNROLL
            if (s[k] > s[best]) best = k;
        }
        out[CLASSES] = best;
        to[c] = out;
    }
}

// N pixels' scores from a stream into a line, so the write is one burst.
template<int N, int CLASSES>
void take_line(hls::stream<word<int32_t, CLASSES> >& from, word<int32_t, CLASSES> line[N]) {
    #pragma HLS INLINE
    take: for (int c = 0; c < N; c++) {
        #pragma HLS PIPELINE II=1
        line[c] = from.read();
    }
}

// Where patch p (in raster order) starts in the image: its row and its left column, and how
// many of its columns are inside.
template<int P>
void patch_place(int p, int width, int& top, int& left, int& inside) {
    #pragma HLS INLINE
    const int cols = (width + P - 1) / P;
    top = (p / cols) * P;
    left = (p % cols) * P;
    inside = width - left < P ? width - left : P;
}

// Patch p of the image, z-scored, a burst a patch row, into a map or a stream. Where the
// patch sticks out of the image, the last row and pixel are repeated (as the optimized
// kernels' read_patch_rows_simple).
template<int P, int BANDS, int WORDS_PER_PIXEL, int FIRST_WORD, int FRAC, int IN_MAX, typename B_T,
         const int32_t (&A)[(WORDS_PER_PIXEL - FIRST_WORD) * 8], const B_T (&B)[(WORDS_PER_PIXEL - FIRST_WORD) * 8],
         typename T>
void read_zscore(const cube_word_t* din, int p, int height, int width, word<T, BANDS> patch[P][P]) {
    int top, left, inside;
    patch_place<P>(p, width, top, left, inside);
    rows: for (int r = 0; r < P; r++) {
        const int y = top + r < height ? top + r : height - 1;
        word<T, BANDS>* row = patch[r];
        read_line<P, BANDS, WORDS_PER_PIXEL, FIRST_WORD, FRAC, IN_MAX, B_T, A, B, T>(
            din + ((int64_t)y * width + left) * WORDS_PER_PIXEL, inside, row);
    }
}
template<int P, int BANDS, int WORDS_PER_PIXEL, int FIRST_WORD, int FRAC, int IN_MAX, typename B_T,
         const int32_t (&A)[(WORDS_PER_PIXEL - FIRST_WORD) * 8], const B_T (&B)[(WORDS_PER_PIXEL - FIRST_WORD) * 8],
         typename T>
void read_zscore(const cube_word_t* din, int p, int height, int width, hls::stream<word<T, BANDS> >& patch) {
    int top, left, inside;
    patch_place<P>(p, width, top, left, inside);
    rows: for (int r = 0; r < P; r++) {
        const int y = top + r < height ? top + r : height - 1;
        read_line<P, BANDS, WORDS_PER_PIXEL, FIRST_WORD, FRAC, IN_MAX, B_T, A, B, T>(
            din + ((int64_t)y * width + left) * WORDS_PER_PIXEL, inside, patch);
    }
}

// Patch p's scores to DDR, the pixels inside the image only (the others are read and dropped).
template<int P, int CLASSES>
void write_scores(hls::stream<word<int32_t, CLASSES> >& scores, score_word_t* dout, int p, int height, int width) {
    int top, left, inside;
    patch_place<P>(p, width, top, left, inside);
    rows: for (int r = 0; r < P; r++) {
        word<int32_t, CLASSES> line[P];
        take_line<P, CLASSES>(scores, line);
        if (top + r < height)
            write_line<P, CLASSES>(line, dout + (int64_t)(top + r) * width + left, inside);
    }
}

// The 1D networks: block b of N consecutive pixels (the last one may be short), z-scored.
template<int N, int BANDS, int WORDS_PER_PIXEL, int FIRST_WORD, int FRAC, int IN_MAX, typename B_T,
         const int32_t (&A)[(WORDS_PER_PIXEL - FIRST_WORD) * 8], const B_T (&B)[(WORDS_PER_PIXEL - FIRST_WORD) * 8],
         typename T>
void read_block(const cube_word_t* din, int b, int pixels, hls::stream<word<T, BANDS> >& block) {
    const int inside = pixels - b * N < N ? pixels - b * N : N;
    read_line<N, BANDS, WORDS_PER_PIXEL, FIRST_WORD, FRAC, IN_MAX, B_T, A, B, T>(
        din + (int64_t)b * N * WORDS_PER_PIXEL, inside, block);
}

// Block b's scores to DDR, the pixels of the image only.
template<int N, int CLASSES>
void write_block(hls::stream<word<int32_t, CLASSES> >& scores, score_word_t* dout, int b, int pixels) {
    const int inside = pixels - b * N < N ? pixels - b * N : N;
    word<int32_t, CLASSES> line[N];
    take_line<N, CLASSES>(scores, line);
    write_line<N, CLASSES>(line, dout + (int64_t)b * N, inside);
}

// The grid of P x P patches over the image; the last row and column may stick out.
template<int P>
int patch_count(int height, int width) {
    return ((height + P - 1) / P) * ((width + P - 1) / P);
}

// The blocks of N pixels over the image; the last may be short.
template<int N>
int block_count(int pixels) {
    return (pixels + N - 1) / N;
}

}  // namespace simple
