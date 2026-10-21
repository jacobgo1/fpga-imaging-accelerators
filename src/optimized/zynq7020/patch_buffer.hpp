#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../shared/patch_buffer.hpp"

// patch_buffer for the Zynq-7020 and Zynq-7030: the z-score, then a ping-pong
// buffer between PORTS read_rows_ahead and the first layer (conv3x3_wide). The
// same as zcu104/patch_buffer.hpp (several words per chunk), in block RAM: the
// 7-series Zynqs have no UltraRAM. With 2 ports and 2 words per chunk that is
// 4 banks of 14 chunks x 16 rows x 32 pixels x 2 buffers x 32 bits: 56 of the
// 7020's 140 BRAM36. See ../shared/patch_buffer.hpp for the round loop.
template<int PH, int PW, int WORDS, int FIRST, int LANES, int P, int PORTS, int FRAC, int IN_MAX,
         const int32_t (&ZA)[(WORDS - FIRST) * LANES], const int32_t (&ZB)[(WORDS - FIRST) * LANES],
         bool DSP, typename WORD_T>
void patch_buffer(hls::stream<WORD_T> raw[PORTS], hls::stream<vec_t<act8_t, P> >& out,
                  int height, int width) {
    static_assert(FIRST >= 0 && FIRST < WORDS, "at least one word of a pixel is kept");
    static_assert(P % LANES == 0, "a chunk is whole words");
    static_assert(PH % PORTS == 0, "every port reads as many rows of a patch");
    static_assert(IN_MAX <= 127, "the z-scores are int8");
    constexpr int KEPT = WORDS - FIRST;                         // kept words per pixel
    constexpr int PER_CHUNK = P / LANES;                        // words per chunk
    constexpr int CHUNKS = (KEPT + PER_CHUNK - 1) / PER_CHUNK;
    constexpr int ROWS = PH / PORTS;                            // rows per port per patch
    // One bank: both ping-pong buffers, [buffer][chunk][row / PORTS][x],
    // flattened. LANES z-scored bands per entry.
    constexpr int BANK_HALF = CHUNKS * ROWS * PW;
    typedef vec_t<act8_t, LANES> word_t;

    // PORTS x PER_CHUNK banks: [row % PORTS = port][word of the chunk]. A cycle
    // writes one word from each port (PORTS rows, the same word) and reads one
    // row's words of a chunk: always different banks. The ping-pong buffer is
    // the high half of the address, not a bank of its own. Each bank has
    // exactly one write and one read in the loop below, at fixed places: a bank
    // whose write or read HLS has to pick at run time gets one store per
    // possible pick, and its single write port then takes that many cycles
    // (II = 5 when the buffer and the word half were array indices).
    word_t bank[PORTS][PER_CHUNK][2 * BANK_HALF];
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=2
    #pragma HLS AGGREGATE variable=bank compact=bit
    #pragma HLS BIND_STORAGE variable=bank type=ram_2p impl=bram

    patch_buffer_run<PH, PW, WORDS, FIRST, LANES, P, PORTS, FRAC, IN_MAX, ZA, ZB, DSP>(
        bank, raw, out, height, width);
}
