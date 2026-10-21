#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// Shared logic of patch_buffer (used by zcu104/patch_buffer.hpp in UltraRAM
// and zynq7020/patch_buffer.hpp in block RAM):
//
// In: from reader r, rows r, r + PORTS, ... of each patch, WORDS words of
// LANES raw uint16 bands per pixel. Words FIRST .. WORDS - 1 are kept, the
// ones before them dropped. Kept band k becomes
//     round((dn - mean) / std * 2^F) = (dn * ZA[k] + ZB[k]) >> FRAC,
// saturated at +-IN_MAX (127: never -128, which conv3x3_wide's packing needs).
// DSP: its PORTS x LANES z-score multipliers in DSPs, or in LUTs.
// Out: P kept bands of a pixel per cycle, in conv3x3_wide's order: chunk by
// chunk, each in raster order. The last chunk's bands past the kept ones are
// zero.
//
// Round k fills patch k into one buffer while it sends patch k - 1 from the
// other; then the buffers swap. n patches take n + 1 rounds of
// max(FILL, EMIT) cycles: the words of one patch per port, or the chunks
// times the pixels.
template<int FRAC, int IN_MAX, bool DSP>
act8_t z_score(uint16_t dn, int32_t a, int32_t b) {
    #pragma HLS INLINE
    const int64_t s = (mul<DSP, int64_t>(dn, a) + b) >> FRAC;
    return (act8_t)(s > IN_MAX ? IN_MAX : s < -IN_MAX ? -IN_MAX : s);
}

template<int PH, int PW, int WORDS, int FIRST, int LANES, int P, int PORTS, int FRAC, int IN_MAX,
         const int32_t (&ZA)[(WORDS - FIRST) * LANES], const int32_t (&ZB)[(WORDS - FIRST) * LANES],
         bool DSP, typename WORD_T>
void patch_buffer_run(
    vec_t<act8_t, LANES> bank[PORTS][P / LANES][2 * (((WORDS - FIRST) + (P / LANES) - 1) / (P / LANES)) * (PH / PORTS) * PW],
    hls::stream<WORD_T> raw[PORTS], hls::stream<vec_t<act8_t, P> >& out,
    int height, int width) {
    #pragma HLS INLINE
    constexpr int KEPT = WORDS - FIRST;                         // kept words per pixel
    constexpr int PER_CHUNK = P / LANES;                        // words per chunk
    constexpr int CHUNKS = (KEPT + PER_CHUNK - 1) / PER_CHUNK;
    constexpr int ROWS = PH / PORTS;                            // rows per port per patch
    constexpr int FILL = ROWS * PW * WORDS;                     // words per port per patch
    constexpr int EMIT = CHUNKS * PH * PW;                      // vectors per patch out
    constexpr int ROUND = FILL > EMIT ? FILL : EMIT;
    constexpr int BANK_HALF = CHUNKS * ROWS * PW;
    typedef vec_t<act8_t, LANES> word_t;

    // The z-score constants of the kept words, one small ROM per lane.
    int32_t za[KEPT][LANES], zb[KEPT][LANES];
    #pragma HLS ARRAY_PARTITION variable=za type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=zb type=complete dim=2
    load_z: for (int k = 0; k < KEPT * LANES; k++) {
        #pragma HLS PIPELINE II=1
        za[k / LANES][k % LANES] = ZA[k];
        zb[k / LANES][k % LANES] = ZB[k];
    }

    const int n = patch_count<PH, PW>(height, width);
    int wb = 0;                                 // the buffer being filled
    round: for (int k = 0; k <= n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=2 max=666
        const bool fill = k < n, emit = k > 0;
        int fr = 0, fx = 0, fw = 0;             // fill: row of the port, pixel, word
        int ec = 0, ey = 0, ex = 0;             // emit: chunk, row, pixel
        step: for (int t = 0; t < ROUND; t++) {
            #pragma HLS PIPELINE II=1
            // Writes go to one half of each bank, reads to the other.
            #pragma HLS DEPENDENCE variable=bank type=inter false
            #pragma HLS DEPENDENCE variable=bank type=intra false
            const bool take = fill && t < FILL;
            WORD_T w[PORTS];
            for (int r = 0; r < PORTS; r++) {
                #pragma HLS UNROLL
                if (take) w[r] = raw[r].read();
            }
            // The words before FIRST are dropped; kept word kw goes to chunk
            // kw / PER_CHUNK, into the bank of its place in the chunk.
            const int kw = fw >= FIRST ? fw - FIRST : 0;
            const bool store = take && fw >= FIRST;
            const int waddr = (wb ? BANK_HALF : 0) + ((kw / PER_CHUNK) * ROWS + fr) * PW + fx;
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
                else { fw = 0; if (fx < PW - 1) fx++; else { fx = 0; fr++; } }
            }

            if (emit && t < EMIT) {
                // All banks are read; the row's port picks its PER_CHUNK. The
                // last chunk's words past the kept ones are never written: send
                // zeros, not whatever the memory holds.
                const int raddr = (wb ? 0 : BANK_HALF) + (ec * ROWS + ey / PORTS) * PW + ex;
                vec_t<act8_t, P> v;
                for (int q = 0; q < PER_CHUNK; q++) {
                    #pragma HLS UNROLL
                    word_t rd[PORTS];
                    for (int r = 0; r < PORTS; r++) {
                        #pragma HLS UNROLL
                        rd[r] = bank[r][q][raddr];
                    }
                    const word_t pick = rd[ey % PORTS];
                    const bool kept = ec * PER_CHUNK + q < KEPT;
                    for (int p = 0; p < LANES; p++) {
                        #pragma HLS UNROLL
                        v.v[q * LANES + p] = kept ? pick.v[p] : (act8_t)0;
                    }
                }
                out.write(v);
                if (ex < PW - 1) ex++;
                else { ex = 0; if (ey < PH - 1) ey++; else { ey = 0; ec++; } }
            }
        }
        wb = 1 - wb;
    }
}
