#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../shared/fixed_point.hpp"
#include "../shared/patch_grid.hpp"

// patch_buffer for the Zynq-7030: the z-score, then a ping-pong buffer between
// PORTS read_rows and the first layer (zynq7030/conv3x3_wide), in block RAM
// (the 7030 has no UltraRAM). The same job as zcu104/patch_buffer.hpp, for a
// first layer that takes one word's bands per cycle: a chunk is one word, so
// each port's rows need one bank, not one per word of a chunk.
//
// In: from reader r, rows r, r + PORTS, ... of each patch, WORDS words of
// LANES raw uint16 bands per pixel. Words FIRST .. WORDS - 1 are kept, the
// ones before them dropped. Kept band k becomes
//     round((dn - mean) / std * 2^F) = (dn * ZA[k] + ZB[k]) >> FRAC,
// saturated at +-IN_MAX.
// DSP: its PORTS x LANES z-score multipliers in DSPs, or in LUTs.
// Out: one kept word (LANES bands) of a pixel per cycle, in conv3x3_wide's
// order: word by word, each in raster order.
//
// Round k fills patch k into one buffer while it sends patch k - 1 from the
// other; then the buffers swap. n patches take n + 1 rounds of
// max(FILL, EMIT) cycles: the words of one patch per port, or the kept words
// times the pixels.
template<int FRAC, int IN_MAX, bool DSP>
act8_t z_score(uint16_t dn, int32_t a, int32_t b) {
    #pragma HLS INLINE
    const int64_t s = (mul<DSP, int64_t>(dn, a) + b) >> FRAC;
    return (act8_t)(s > IN_MAX ? IN_MAX : s < -IN_MAX ? -IN_MAX : s);
}

template<int PH, int PW, int WORDS, int FIRST, int LANES, int PORTS, int FRAC, int IN_MAX,
         const int32_t (&ZA)[(WORDS - FIRST) * LANES], const int32_t (&ZB)[(WORDS - FIRST) * LANES],
         bool DSP, typename WORD_T>
void patch_buffer(hls::stream<WORD_T> raw[PORTS], hls::stream<vec_t<act8_t, LANES> >& out,
                  int height, int width) {
    static_assert(FIRST >= 0 && FIRST < WORDS, "at least one word of a pixel is kept");
    static_assert(PH % PORTS == 0, "every port reads as many rows of a patch");
    static_assert(IN_MAX <= 127, "the z-scores are int8");
    constexpr int KEPT = WORDS - FIRST;                         // kept words per pixel
    constexpr int ROWS = PH / PORTS;                            // rows per port per patch
    constexpr int FILL = ROWS * PW * WORDS;                     // words per port per patch
    constexpr int EMIT = KEPT * PH * PW;                        // words per patch out
    constexpr int ROUND = FILL > EMIT ? FILL : EMIT;
    // One bank per port: both ping-pong buffers, [buffer][word][row / PORTS][x],
    // flattened. LANES z-scored bands per entry.
    constexpr int BANK_HALF = KEPT * ROWS * PW;
    typedef vec_t<act8_t, LANES> word_t;

    // A cycle writes one word into each bank and reads one row's word from
    // all of them, keeping its port's: one write and one read per bank, at
    // fixed places (a bank picked at run time would get one store per pick).
    // The ping-pong buffer is the high half of the address.
    word_t bank[PORTS][2 * BANK_HALF];
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=1
    #pragma HLS AGGREGATE variable=bank compact=bit
    #pragma HLS BIND_STORAGE variable=bank type=ram_2p impl=bram
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
        int ew = 0, ey = 0, ex = 0;             // emit: kept word, row, pixel
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
            // The words before FIRST are dropped.
            const int kw = fw >= FIRST ? fw - FIRST : 0;
            const bool store = take && fw >= FIRST;
            const int waddr = (wb ? BANK_HALF : 0) + (kw * ROWS + fr) * PW + fx;
            for (int r = 0; r < PORTS; r++) {
                #pragma HLS UNROLL
                word_t z;
                for (int p = 0; p < LANES; p++) {
                    #pragma HLS UNROLL
                    z.v[p] = z_score<FRAC, IN_MAX, DSP>(w[r][p], za[kw][p], zb[kw][p]);
                }
                if (store) bank[r][waddr] = z;
            }
            if (take) {
                if (fw < WORDS - 1) fw++;
                else { fw = 0; if (fx < PW - 1) fx++; else { fx = 0; fr++; } }
            }

            if (emit && t < EMIT) {
                const int raddr = (wb ? 0 : BANK_HALF) + (ew * ROWS + ey / PORTS) * PW + ex;
                word_t rd[PORTS];
                for (int r = 0; r < PORTS; r++) {
                    #pragma HLS UNROLL
                    rd[r] = bank[r][raddr];
                }
                out.write(rd[ey % PORTS]);
                if (ex < PW - 1) ex++;
                else { ex = 0; if (ey < PH - 1) ey++; else { ey = 0; ew++; } }
            }
        }
        wb = 1 - wb;
    }
}
