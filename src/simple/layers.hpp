#pragma once

// The simple kernels' layers: the golden kernels' plain loops (src/golden/layers.hpp) in
// integers, made fast with pragmas only. A kernel calls them once per patch inside a
// DATAFLOW region, so the arrays between them become ping-pong buffers and every
// layer works on a different patch at the same time.
//
// A feature map is [row][col] of pixel words, all of a pixel's channels in one
// hls::vector: one memory as wide as the pixel, so no array needs partitioning and
// it takes only the bits it holds. The arithmetic is the optimized kernels', so the
// answers are the same bit for bit: exact int32 sums, the bias shifted up to the sums'
// fraction bits, then rounded half up, ReLU and saturation (or the plain rounded sums
// for the scores).

#include <stdint.h>
#include "../common/hls_vector_shim.hpp"

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

typedef hls::vector<uint16_t, 8> cube_word_t;   // 8 raw bands, 128 bits
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// sum * 2^-SHIFT rounded half up (a negative SHIFT shifts left, exactly); with RELU,
// then clamped to 0 .. ACT_MAX.
template<bool RELU, int ACT_MAX, int SHIFT>
int32_t finish(int32_t sum) {
    #pragma HLS INLINE
    const int64_t rounded = SHIFT <= 0 ? (int64_t)sum * ((int64_t)1 << (SHIFT < 0 ? -SHIFT : 0))
                                       : ((int64_t)sum + ((int64_t)1 << (SHIFT > 0 ? SHIFT - 1 : 0))) >> SHIFT;
    if (!RELU) return (int32_t)rounded;
    return rounded < 0 ? 0 : rounded > ACT_MAX ? ACT_MAX : (int32_t)rounded;
}

// Conv2d, same padding, K = 1 or 3: OUT channels from IN_A channels of a, then IN_B of b
// (torch.cat([a, b], dim=1); IN_B = 0 for one input, b is then never read). UPSAMPLED:
// the inputs are half the size and read at (y/2, x/2), nearest-neighbour upsampling.
//
// One pipelined loop over pixel, output chunk, tap and input chunk: UI inputs times
// UO outputs a step, H*W*K*K*(IN/UI)*(OUT/UO) steps a patch. The taps are not
// unrolled: a step reads one input pixel, so there is no 3x3 window to build.
template<int H, int W, int IN_A, int IN_B, int OUT, int K, int UI, int UO, bool RELU, int ACT_MAX,
         int BIAS_SHIFT, int SHIFT, bool UPSAMPLED,
         const int8_t (&WEIGHT)[OUT][IN_A + IN_B][K][K], const int8_t (&BIAS)[OUT],
         typename TI, typename TO, typename B_WORD>
void conv_body(const word<TI, IN_A> a[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W],
               const B_WORD b[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W],
               word<TO, OUT> out[H][W]) {
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
    word<TO, OUT> result;
    int row = 0, col = 0, oc = 0, tap = 0, ic = 0;
    steps: for (int s = 0; s < H * W * STEPS; s++) {
        #pragma HLS PIPELINE II=1
        const int y = row + tap / K - PAD, x = col + tap % K - PAD;
        const bool inside = y >= 0 && y < H && x >= 0 && x < W;
        const int sy = inside ? (UPSAMPLED ? y / 2 : y) : 0, sx = inside ? (UPSAMPLED ? x / 2 : x) : 0;

        // This step's UI inputs; a border tap reads zeros.
        TI v[UI];
        #pragma HLS ARRAY_PARTITION variable=v type=complete
        const word<TI, IN_A> pa = a[sy][sx];
        B_WORD pb;
        if (IN_B > 0) pb = b[sy][sx];
        for (int i = 0; i < UI; i++) {
            #pragma HLS UNROLL
            TI value = 0;
            // Each chunk at a fixed place: the run-time choice is which assignment runs.
            for (int g = 0; g < IC; g++) {
                #pragma HLS UNROLL
                if (ic == g) value = g < IC_A ? pa[g * UI + i] : pb[(IN_B > 0 ? g - IC_A : 0) * UI + i];
            }
            v[i] = inside ? value : (TI)0;
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
            for (int g = 0; g < OC; g++) {
                #pragma HLS UNROLL
                if (last && oc == g) result[g * UO + j] = (TO)finish<RELU, ACT_MAX, SHIFT>(sum);
            }
        }
        if (last && oc == OC - 1) out[row][col] = result;

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

// The usual conv, one input (passed as b too, never read there).
template<int H, int W, int IN, int OUT, int K, int UI, int UO, bool RELU, int ACT_MAX,
         int BIAS_SHIFT, int SHIFT, bool UPSAMPLED,
         const int8_t (&WEIGHT)[OUT][IN][K][K], const int8_t (&BIAS)[OUT], typename TI, typename TO>
void conv(const word<TI, IN> in[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W], word<TO, OUT> out[H][W]) {
    conv_body<H, W, IN, 0, OUT, K, UI, UO, RELU, ACT_MAX, BIAS_SHIFT, SHIFT, UPSAMPLED, WEIGHT, BIAS,
              TI, TO, word<TI, IN> >(in, in, out);
}

// The conv after a concat: a's channels, then b's.
template<int H, int W, int IN_A, int IN_B, int OUT, int K, int UI, int UO, bool RELU, int ACT_MAX,
         int BIAS_SHIFT, int SHIFT, bool UPSAMPLED,
         const int8_t (&WEIGHT)[OUT][IN_A + IN_B][K][K], const int8_t (&BIAS)[OUT], typename TI, typename TO>
void conv_concat(const word<TI, IN_A> a[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W],
                 const word<TI, IN_B> b[UPSAMPLED ? H / 2 : H][UPSAMPLED ? W / 2 : W], word<TO, OUT> out[H][W]) {
    conv_body<H, W, IN_A, IN_B, OUT, K, UI, UO, RELU, ACT_MAX, BIAS_SHIFT, SHIFT, UPSAMPLED, WEIGHT, BIAS,
              TI, TO, word<TI, IN_B> >(a, b, out);
}

// 2 x 2 max pooling, stride 2: four reads an output pixel, one a cycle.
template<int H, int W, int C, typename T>
void maxpool2(const word<T, C> in[H][W], word<T, C> out[H / 2][W / 2]) {
    word<T, C> largest;
    pool: for (int s = 0; s < H * W; s++) {
        #pragma HLS PIPELINE II=1
        const int p = s / 4, d = s % 4;
        const int row = p / (W / 2), col = p % (W / 2);
        const word<T, C> v = in[2 * row + d / 2][2 * col + d % 2];
        for (int c = 0; c < C; c++) {
            #pragma HLS UNROLL
            if (d == 0 || v[c] > largest[c]) largest[c] = v[c];
        }
        if (d == 3) out[row][col] = largest;
    }
}

// One map to two: a skip connection's and the next layer's copies.
template<int H, int W, int C, typename T>
void duplicate(const word<T, C> in[H][W], word<T, C> first[H][W], word<T, C> second[H][W]) {
    copy: for (int s = 0; s < H * W; s++) {
        #pragma HLS PIPELINE II=1
        const word<T, C> v = in[s / W][s % W];
        first[s / W][s % W] = v;
        second[s / W][s % W] = v;
    }
}

// INSIDE consecutive pixels of the raw cube from DDR in one burst, z-scored into
// line[0 .. INSIDE - 1]; the rest of the line repeats the last of them. Words before
// FIRST_WORD are dropped; kept band k is (dn * A[k] + B[k]) >> FRAC, saturated to
// +-IN_MAX. B_T: B's type, int32 or int64.
template<int N, int BANDS, int WORDS_PER_PIXEL, int FIRST_WORD, int FRAC, int IN_MAX, typename B_T,
         const int32_t (&A)[(WORDS_PER_PIXEL - FIRST_WORD) * 8], const B_T (&B)[(WORDS_PER_PIXEL - FIRST_WORD) * 8],
         typename T>
void read_line(const cube_word_t* from, int inside, word<T, BANDS> line[N]) {
    #pragma HLS INLINE
    constexpr int KEPT_WORDS = WORDS_PER_PIXEL - FIRST_WORD;
    word<T, BANDS> pixel;
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
                    pixel[band] = (T)(z > IN_MAX ? IN_MAX : z < -IN_MAX ? -IN_MAX : z);
                }
            }
        }
        if (w == WORDS_PER_PIXEL - 1) {
            line[col] = pixel;
            col++;
            w = 0;
        } else {
            w++;
        }
    }
    edge: for (int c = inside; c < N; c++) {
        #pragma HLS PIPELINE II=1
        line[c] = pixel;
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

// Patch p of the image (in raster order), z-scored, a burst a patch row. Where the patch
// sticks out of the image, the last row and pixel are repeated (as the optimized kernels'
// read_patch_rows_simple).
template<int P, int BANDS, int WORDS_PER_PIXEL, int FIRST_WORD, int FRAC, int IN_MAX, typename B_T,
         const int32_t (&A)[(WORDS_PER_PIXEL - FIRST_WORD) * 8], const B_T (&B)[(WORDS_PER_PIXEL - FIRST_WORD) * 8],
         typename T>
void read_zscore(const cube_word_t* din, int p, int height, int width, word<T, BANDS> patch[P][P]) {
    const int cols = (width + P - 1) / P, pr = p / cols, left = (p % cols) * P;
    const int inside = width - left < P ? width - left : P;
    rows: for (int r = 0; r < P; r++) {
        const int y = pr * P + r < height ? pr * P + r : height - 1;
        read_line<P, BANDS, WORDS_PER_PIXEL, FIRST_WORD, FRAC, IN_MAX, B_T, A, B>(
            din + ((int64_t)y * width + left) * WORDS_PER_PIXEL, inside, patch[r]);
    }
}

// Patch p's scores to DDR, the pixels inside the image only.
template<int P, int CLASSES>
void write_scores(const word<int32_t, CLASSES> scores[P][P], score_word_t* dout, int p, int height, int width) {
    const int cols = (width + P - 1) / P, pr = p / cols, left = (p % cols) * P;
    const int inside = width - left < P ? width - left : P;
    rows: for (int r = 0; r < P; r++) {
        const int y = pr * P + r;
        if (y >= height) break;
        write_line<P, CLASSES>(scores[r], dout + (int64_t)y * width + left, inside);
    }
}

// The 1D networks: block b of N consecutive pixels (the last one may be short), z-scored.
template<int N, int BANDS, int WORDS_PER_PIXEL, int FIRST_WORD, int FRAC, int IN_MAX, typename B_T,
         const int32_t (&A)[(WORDS_PER_PIXEL - FIRST_WORD) * 8], const B_T (&B)[(WORDS_PER_PIXEL - FIRST_WORD) * 8],
         typename T>
void read_block(const cube_word_t* din, int b, int pixels, word<T, BANDS> block[1][N]) {
    const int inside = pixels - b * N < N ? pixels - b * N : N;
    read_line<N, BANDS, WORDS_PER_PIXEL, FIRST_WORD, FRAC, IN_MAX, B_T, A, B>(
        din + (int64_t)b * N * WORDS_PER_PIXEL, inside, block[0]);
}

// Block b's scores to DDR, the pixels of the image only.
template<int N, int CLASSES>
void write_block(const word<int32_t, CLASSES> scores[1][N], score_word_t* dout, int b, int pixels) {
    const int inside = pixels - b * N < N ? pixels - b * N : N;
    write_line<N, CLASSES>(scores[0], dout + (int64_t)b * N, inside);
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
