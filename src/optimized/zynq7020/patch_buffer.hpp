#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../shared/patch_buffer.hpp"

// The patch buffer in block RAM. One bank per read port and word of a chunk, so a
// cycle's writes and its read never meet; the two buffers are the address's halves.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int FIRST_KEPT_WORD, int BANDS_PER_WORD,
         int BANDS_PER_CYCLE, int READ_PORTS, int ZSCORE_SHIFT, int ZSCORE_MAX,
         const int32_t (&ZSCORE_SCALE)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         const int32_t (&ZSCORE_OFFSET)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         bool DSP, typename WORD_T>
void patch_buffer(hls::stream<WORD_T> raw[READ_PORTS], hls::stream<vec_t<act8_t, BANDS_PER_CYCLE> >& out,
                  int height, int width) {
    constexpr int KEPT_WORDS = WORDS_PER_PIXEL - FIRST_KEPT_WORD;
    constexpr int WORDS_PER_CHUNK = BANDS_PER_CYCLE / BANDS_PER_WORD;
    constexpr int CHUNKS = (KEPT_WORDS + WORDS_PER_CHUNK - 1) / WORDS_PER_CHUNK;
    constexpr int BUFFER_SIZE = CHUNKS * (PATCH_HEIGHT / READ_PORTS) * PATCH_WIDTH;

    vec_t<act8_t, BANDS_PER_WORD> bank[READ_PORTS][WORDS_PER_CHUNK][2 * BUFFER_SIZE];
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=2
    #pragma HLS AGGREGATE variable=bank compact=bit
    #pragma HLS BIND_STORAGE variable=bank type=ram_2p impl=bram

    patch_buffer_run<PATCH_HEIGHT, PATCH_WIDTH, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD, BANDS_PER_CYCLE,
                     READ_PORTS, ZSCORE_SHIFT, ZSCORE_MAX, ZSCORE_SCALE, ZSCORE_OFFSET, DSP>(
        bank, raw, out, height, width);
}
