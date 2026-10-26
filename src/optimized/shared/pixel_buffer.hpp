#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "patch_buffer.hpp"

// pixel_buffer: the z-score, then a strip ping-pong buffer between PORTS
// read_rows_ahead and a pixel-major first layer (dwt_proj in WaveletCNN).
//
// Unlike conv3x3_wide (which wants chunk 0 of every pixel in the patch, then
// chunk 1, ...), a 1D spectral / 1x1 pointwise layer wants all bands of pixel
// (y, x) together before moving to (y, x + 1). Because the PORTS readers fetch
// rows s*PORTS .. s*PORTS + PORTS - 1 of a patch at the same time, a ping-pong
// buffer of just one strip (PORTS rows x PW pixels -- 8x smaller than a full
// patch with 4 ports, 16x smaller with 2 ports) is enough to turn their
// row-interleaved words into raster-ordered pixels in block RAM.
//
// Out: for each pixel in raster order, OUT_CHUNKS vectors of P z-scored int8
// bands. Words past the kept ones (including any extra chunk needed to flush a
// spectral filter's right zero-padding) are emitted as zeros without taking
// space in the buffer.
//
// BLOCKS: the pixels in 2x2-block order instead (wino_conv3x3.hpp's): a strip
// holds PORTS / 2 rows of blocks, and each block's four pixels leave one after
// another, top left, top right, bottom left, bottom right. Only the order the
// buffer is read in changes.
template<int PH, int PW, int WORDS, int FIRST, int LANES, int P, int PORTS, int OUT_CHUNKS,
         int FRAC, int IN_MAX,
         const int32_t (&ZA)[(WORDS - FIRST) * LANES], const int32_t (&ZB)[(WORDS - FIRST) * LANES],
         bool DSP, typename WORD_T, bool BLOCKS = false>
void pixel_buffer(hls::stream<WORD_T> raw[PORTS], hls::stream<vec_t<act8_t, P> >& out,
                  int height, int width) {
    static_assert(FIRST >= 0 && FIRST < WORDS, "at least one word of a pixel is kept");
    static_assert(P % LANES == 0, "a chunk is whole words");
    static_assert(PH % PORTS == 0, "every port reads as many rows of a patch");
    static_assert(IN_MAX <= 127, "the z-scores are int8");
    static_assert(!BLOCKS || (PORTS % 2 == 0 && PW % 2 == 0), "a strip of whole 2x2 blocks");
    constexpr int KEPT = WORDS - FIRST;                         // kept words per pixel
    constexpr int PER_CHUNK = P / LANES;                        // words per chunk
    constexpr int MEM_CHUNKS = (KEPT + PER_CHUNK - 1) / PER_CHUNK;
    static_assert(OUT_CHUNKS >= MEM_CHUNKS, "OUT_CHUNKS must cover all kept words");
    constexpr int STRIPS_PER_PATCH = PH / PORTS;                // PORTS-row strips per patch
    constexpr int FILL = PW * WORDS;                            // words per port per strip
    constexpr int EMIT = PORTS * PW * OUT_CHUNKS;               // vectors per strip out
    constexpr int ROUND = FILL > EMIT ? FILL : EMIT;
    // One bank: both ping-pong strip buffers, [buffer][chunk][x], flattened.
    constexpr int BANK_HALF = MEM_CHUNKS * PW;
    typedef vec_t<act8_t, LANES> word_t;

    word_t bank[PORTS][PER_CHUNK][2 * BANK_HALF];
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=2
    #pragma HLS AGGREGATE variable=bank compact=bit
    #pragma HLS BIND_STORAGE variable=bank type=ram_2p impl=bram

    int32_t za[KEPT][LANES], zb[KEPT][LANES];
    #pragma HLS ARRAY_PARTITION variable=za type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=zb type=complete dim=2
    load_z: for (int k = 0; k < KEPT * LANES; k++) {
        #pragma HLS PIPELINE II=1
        za[k / LANES][k % LANES] = ZA[k];
        zb[k / LANES][k % LANES] = ZB[k];
    }

    const int n_strips = patch_count<PH, PW>(height, width) * STRIPS_PER_PATCH;
    int wb = 0;                                 // the strip buffer being filled
    strip: for (int k = 0; k <= n_strips; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=9 max=5321
        const bool fill = k < n_strips, emit = k > 0;
        int fx = 0, fw = 0;                     // fill: pixel in the row, word of the pixel
        int er = 0, ex = 0, ec = 0;             // emit: row in the strip, pixel, chunk
        int eq = 0;                             // BLOCKS: the pixel of the block
        step: for (int t = 0; t < ROUND; t++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=bank type=inter false
            #pragma HLS DEPENDENCE variable=bank type=intra false
            const bool take = fill && t < FILL;
            WORD_T w[PORTS];
            for (int r = 0; r < PORTS; r++) {
                #pragma HLS UNROLL
                if (take) w[r] = raw[r].read();
            }
            const int kw = fw >= FIRST ? fw - FIRST : 0;
            const bool store = take && fw >= FIRST;
            const int waddr = (wb ? BANK_HALF : 0) + (kw / PER_CHUNK) * PW + fx;
            for (int r = 0; r < PORTS; r++) {
                #pragma HLS UNROLL
                word_t z;
                for (int p = 0; p < LANES; p++) {
                    #pragma HLS UNROLL
                    z.v[p] = z_score<FRAC, IN_MAX, DSP>(w[r][p], za[kw][p], zb[kw][p]);
                }
                for (int q = 0; q < PER_CHUNK; q++) {
                    #pragma HLS UNROLL
                    if (store && kw % PER_CHUNK == q) bank[r][q][waddr] = z;
                }
            }
            if (take) {
                if (fw < WORDS - 1) fw++;
                else { fw = 0; fx++; }
            }

            if (emit && t < EMIT) {
                // BLOCKS: er and ex count blocks; the pixel is eq's corner of them.
                const int row = BLOCKS ? 2 * er + eq / 2 : er;
                const int px = BLOCKS ? 2 * ex + eq % 2 : ex;
                const int mem_c = ec < MEM_CHUNKS ? ec : 0;
                const int raddr = (wb ? 0 : BANK_HALF) + mem_c * PW + px;
                vec_t<act8_t, P> v;
                for (int q = 0; q < PER_CHUNK; q++) {
                    #pragma HLS UNROLL
                    word_t rd[PORTS];
                    for (int r = 0; r < PORTS; r++) {
                        #pragma HLS UNROLL
                        rd[r] = bank[r][q][raddr];
                    }
                    const word_t pick = rd[row];
                    const bool kept = ec * PER_CHUNK + q < KEPT;
                    for (int p = 0; p < LANES; p++) {
                        #pragma HLS UNROLL
                        v.v[q * LANES + p] = kept ? pick.v[p] : (act8_t)0;
                    }
                }
                out.write(v);
                if (ec < OUT_CHUNKS - 1) ec++;
                else if (BLOCKS && eq < 3) { ec = 0; eq++; }
                else { ec = 0; eq = 0; if (ex < (BLOCKS ? PW / 2 : PW) - 1) ex++; else { ex = 0; er++; } }
            }
        }
        wb = 1 - wb;
    }
}
