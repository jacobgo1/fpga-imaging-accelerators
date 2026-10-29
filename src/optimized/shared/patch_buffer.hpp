#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// The z-score and the ping-pong buffer between the readers and the first layer;
// zcu104/ keeps it in UltraRAM, zynq7020/ in block RAM.
//
// Reader r sends rows r, r + READ_PORTS, ... of each patch, WORDS_PER_PIXEL words
// a pixel. Words before FIRST_KEPT_WORD are dropped; kept band k becomes
// (raw * ZSCORE_SCALE[k] + ZSCORE_OFFSET[k]) >> ZSCORE_SHIFT, saturated at
// +-ZSCORE_MAX. Out go BANDS_PER_CYCLE bands of a pixel a cycle, chunk by chunk,
// each chunk in raster order; bands past the last kept one are zero.
// While one patch fills one buffer, the previous one is sent from the other.
template<int ZSCORE_SHIFT, int ZSCORE_MAX, bool DSP>
act8_t z_score(uint16_t raw, int32_t scale, int32_t offset) {
    #pragma HLS INLINE
    const int64_t z = (mul<DSP, int64_t>(raw, scale) + offset) >> ZSCORE_SHIFT;
    return (act8_t)(z > ZSCORE_MAX ? ZSCORE_MAX : z < -ZSCORE_MAX ? -ZSCORE_MAX : z);
}

template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int FIRST_KEPT_WORD, int BANDS_PER_WORD,
         int BANDS_PER_CYCLE, int READ_PORTS, int ZSCORE_SHIFT, int ZSCORE_MAX,
         const int32_t (&ZSCORE_SCALE)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         const int32_t (&ZSCORE_OFFSET)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         bool DSP, typename WORD_T>
void patch_buffer_run(
    vec_t<act8_t, BANDS_PER_WORD> bank[READ_PORTS][BANDS_PER_CYCLE / BANDS_PER_WORD]
        [2 * (((WORDS_PER_PIXEL - FIRST_KEPT_WORD) + (BANDS_PER_CYCLE / BANDS_PER_WORD) - 1)
              / (BANDS_PER_CYCLE / BANDS_PER_WORD)) * (PATCH_HEIGHT / READ_PORTS) * PATCH_WIDTH],
    hls::stream<WORD_T> raw[READ_PORTS], hls::stream<vec_t<act8_t, BANDS_PER_CYCLE> >& out,
    int height, int width) {
    #pragma HLS INLINE
    constexpr int KEPT_WORDS = WORDS_PER_PIXEL - FIRST_KEPT_WORD;
    constexpr int WORDS_PER_CHUNK = BANDS_PER_CYCLE / BANDS_PER_WORD;
    constexpr int CHUNKS = (KEPT_WORDS + WORDS_PER_CHUNK - 1) / WORDS_PER_CHUNK;
    constexpr int ROWS_PER_PORT = PATCH_HEIGHT / READ_PORTS;
    constexpr int FILL_STEPS = ROWS_PER_PORT * PATCH_WIDTH * WORDS_PER_PIXEL;
    constexpr int EMIT_STEPS = CHUNKS * PATCH_HEIGHT * PATCH_WIDTH;
    constexpr int ROUND_STEPS = FILL_STEPS > EMIT_STEPS ? FILL_STEPS : EMIT_STEPS;
    constexpr int BUFFER_SIZE = CHUNKS * ROWS_PER_PORT * PATCH_WIDTH;
    typedef vec_t<act8_t, BANDS_PER_WORD> word_t;

    int32_t scale[KEPT_WORDS][BANDS_PER_WORD], offset[KEPT_WORDS][BANDS_PER_WORD];
    #pragma HLS ARRAY_PARTITION variable=scale type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=offset type=complete dim=2
    load_z: for (int k = 0; k < KEPT_WORDS * BANDS_PER_WORD; k++) {
        #pragma HLS PIPELINE II=1
        scale[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_SCALE[k];
        offset[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_OFFSET[k];
    }

    const int patches = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width);
    int filling_second = 0;
    round: for (int patch = 0; patch <= patches; patch++) {
        #pragma HLS LOOP_TRIPCOUNT min=2 max=666
        const bool fill = patch < patches, emit = patch > 0;
        int fill_row = 0, fill_col = 0, fill_word = 0;
        int emit_chunk = 0, emit_row = 0, emit_col = 0;
        step: for (int t = 0; t < ROUND_STEPS; t++) {
            #pragma HLS PIPELINE II=1
            // Writes go to one half of each bank, reads to the other.
            #pragma HLS DEPENDENCE variable=bank type=inter false
            #pragma HLS DEPENDENCE variable=bank type=intra false
            const bool take = fill && t < FILL_STEPS;
            WORD_T words[READ_PORTS];
            for (int port = 0; port < READ_PORTS; port++) {
                #pragma HLS UNROLL
                if (take) words[port] = raw[port].read();
            }
            const int kept_word = fill_word >= FIRST_KEPT_WORD ? fill_word - FIRST_KEPT_WORD : 0;
            const bool store = take && fill_word >= FIRST_KEPT_WORD;
            const int write_address = (filling_second ? BUFFER_SIZE : 0)
                                    + ((kept_word / WORDS_PER_CHUNK) * ROWS_PER_PORT + fill_row) * PATCH_WIDTH + fill_col;
            for (int port = 0; port < READ_PORTS; port++) {
                #pragma HLS UNROLL
                word_t zscores;
                for (int b = 0; b < BANDS_PER_WORD; b++) {
                    #pragma HLS UNROLL
                    zscores.v[b] = z_score<ZSCORE_SHIFT, ZSCORE_MAX, DSP>(
                        words[port][b], scale[kept_word][b], offset[kept_word][b]);
                }
                // Each bank is written at one fixed place: a bank chosen at run
                // time would be duplicated per choice (II = 5).
                for (int w = 0; w < WORDS_PER_CHUNK; w++) {
                    #pragma HLS UNROLL
                    if (store && kept_word % WORDS_PER_CHUNK == w) bank[port][w][write_address] = zscores;
                }
            }
            if (take) {
                if (fill_word < WORDS_PER_PIXEL - 1) fill_word++;
                else { fill_word = 0; if (fill_col < PATCH_WIDTH - 1) fill_col++; else { fill_col = 0; fill_row++; } }
            }

            if (emit && t < EMIT_STEPS) {
                const int read_address = (filling_second ? 0 : BUFFER_SIZE)
                                       + (emit_chunk * ROWS_PER_PORT + emit_row / READ_PORTS) * PATCH_WIDTH + emit_col;
                vec_t<act8_t, BANDS_PER_CYCLE> chunk;
                for (int w = 0; w < WORDS_PER_CHUNK; w++) {
                    #pragma HLS UNROLL
                    word_t from_port[READ_PORTS];
                    for (int port = 0; port < READ_PORTS; port++) {
                        #pragma HLS UNROLL
                        from_port[port] = bank[port][w][read_address];
                    }
                    const word_t picked = from_port[emit_row % READ_PORTS];
                    // The words past the kept ones were never written.
                    const bool kept = emit_chunk * WORDS_PER_CHUNK + w < KEPT_WORDS;
                    for (int b = 0; b < BANDS_PER_WORD; b++) {
                        #pragma HLS UNROLL
                        chunk.v[w * BANDS_PER_WORD + b] = kept ? picked.v[b] : (act8_t)0;
                    }
                }
                out.write(chunk);
                if (emit_col < PATCH_WIDTH - 1) emit_col++;
                else { emit_col = 0; if (emit_row < PATCH_HEIGHT - 1) emit_row++; else { emit_row = 0; emit_chunk++; } }
            }
        }
        filling_second = 1 - filling_second;
    }
}
