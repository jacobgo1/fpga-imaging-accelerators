#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "patch_buffer.hpp"

// The z-score and a ping-pong buffer of one strip (READ_PORTS rows of a patch),
// for a first layer that wants all bands of a pixel before the next pixel. The
// readers fetch a strip's rows at once, so one strip is enough to put their
// words in raster order. Out: each pixel's OUT_CHUNKS chunks of BANDS_PER_CYCLE
// bands; chunks past the kept words are zeros and take no memory.
// BLOCKS: the pixels in 2x2-block order instead (wino_conv3x3.hpp), top left,
// top right, bottom left, bottom right; only the read order changes.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int WORDS_PER_PIXEL, int FIRST_KEPT_WORD, int BANDS_PER_WORD,
         int BANDS_PER_CYCLE, int READ_PORTS, int OUT_CHUNKS, int ZSCORE_SHIFT, int ZSCORE_MAX,
         const int32_t (&ZSCORE_SCALE)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         const int32_t (&ZSCORE_OFFSET)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         bool DSP, typename WORD_T, bool BLOCKS = false>
void pixel_buffer(hls::stream<WORD_T> raw[READ_PORTS], hls::stream<vec_t<act8_t, BANDS_PER_CYCLE> >& out,
                  int height, int width) {
    constexpr int KEPT_WORDS = WORDS_PER_PIXEL - FIRST_KEPT_WORD;
    constexpr int WORDS_PER_CHUNK = BANDS_PER_CYCLE / BANDS_PER_WORD;
    constexpr int STORED_CHUNKS = (KEPT_WORDS + WORDS_PER_CHUNK - 1) / WORDS_PER_CHUNK;
    constexpr int STRIPS_PER_PATCH = PATCH_HEIGHT / READ_PORTS;
    constexpr int FILL_STEPS = PATCH_WIDTH * WORDS_PER_PIXEL;
    constexpr int EMIT_STEPS = READ_PORTS * PATCH_WIDTH * OUT_CHUNKS;
    constexpr int ROUND_STEPS = FILL_STEPS > EMIT_STEPS ? FILL_STEPS : EMIT_STEPS;
    constexpr int BUFFER_SIZE = STORED_CHUNKS * PATCH_WIDTH;
    typedef vec_t<act8_t, BANDS_PER_WORD> word_t;

    word_t bank[READ_PORTS][WORDS_PER_CHUNK][2 * BUFFER_SIZE];
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=bank type=complete dim=2
    #pragma HLS AGGREGATE variable=bank compact=bit
    #pragma HLS BIND_STORAGE variable=bank type=ram_2p impl=bram

    int32_t scale[KEPT_WORDS][BANDS_PER_WORD], offset[KEPT_WORDS][BANDS_PER_WORD];
    #pragma HLS ARRAY_PARTITION variable=scale type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=offset type=complete dim=2
    load_z: for (int k = 0; k < KEPT_WORDS * BANDS_PER_WORD; k++) {
        #pragma HLS PIPELINE II=1
        scale[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_SCALE[k];
        offset[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_OFFSET[k];
    }

    const int strips = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * STRIPS_PER_PATCH;
    int filling_second = 0;
    strip: for (int s = 0; s <= strips; s++) {
        #pragma HLS LOOP_TRIPCOUNT min=9 max=5321
        const bool fill = s < strips, emit = s > 0;
        int fill_col = 0, fill_word = 0;
        int emit_row = 0, emit_col = 0, emit_chunk = 0;
        int block_pixel = 0;                    // BLOCKS: which pixel of the block
        step: for (int t = 0; t < ROUND_STEPS; t++) {
            #pragma HLS PIPELINE II=1
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
                                    + (kept_word / WORDS_PER_CHUNK) * PATCH_WIDTH + fill_col;
            for (int port = 0; port < READ_PORTS; port++) {
                #pragma HLS UNROLL
                word_t zscores;
                for (int b = 0; b < BANDS_PER_WORD; b++) {
                    #pragma HLS UNROLL
                    zscores.v[b] = z_score<ZSCORE_SHIFT, ZSCORE_MAX, DSP>(
                        words[port][b], scale[kept_word][b], offset[kept_word][b]);
                }
                for (int w = 0; w < WORDS_PER_CHUNK; w++) {
                    #pragma HLS UNROLL
                    if (store && kept_word % WORDS_PER_CHUNK == w) bank[port][w][write_address] = zscores;
                }
            }
            if (take) {
                if (fill_word < WORDS_PER_PIXEL - 1) fill_word++;
                else { fill_word = 0; fill_col++; }
            }

            if (emit && t < EMIT_STEPS) {
                // With BLOCKS, emit_row and emit_col count blocks.
                const int row = BLOCKS ? 2 * emit_row + block_pixel / 2 : emit_row;
                const int col = BLOCKS ? 2 * emit_col + block_pixel % 2 : emit_col;
                const int stored_chunk = emit_chunk < STORED_CHUNKS ? emit_chunk : 0;
                const int read_address = (filling_second ? 0 : BUFFER_SIZE) + stored_chunk * PATCH_WIDTH + col;
                vec_t<act8_t, BANDS_PER_CYCLE> chunk;
                for (int w = 0; w < WORDS_PER_CHUNK; w++) {
                    #pragma HLS UNROLL
                    word_t from_port[READ_PORTS];
                    for (int port = 0; port < READ_PORTS; port++) {
                        #pragma HLS UNROLL
                        from_port[port] = bank[port][w][read_address];
                    }
                    const word_t picked = from_port[row];
                    const bool kept = emit_chunk * WORDS_PER_CHUNK + w < KEPT_WORDS;
                    for (int b = 0; b < BANDS_PER_WORD; b++) {
                        #pragma HLS UNROLL
                        chunk.v[w * BANDS_PER_WORD + b] = kept ? picked.v[b] : (act8_t)0;
                    }
                }
                out.write(chunk);
                if (emit_chunk < OUT_CHUNKS - 1) emit_chunk++;
                else if (BLOCKS && block_pixel < 3) { emit_chunk = 0; block_pixel++; }
                else {
                    emit_chunk = 0; block_pixel = 0;
                    if (emit_col < (BLOCKS ? PATCH_WIDTH / 2 : PATCH_WIDTH) - 1) emit_col++;
                    else { emit_col = 0; emit_row++; }
                }
            }
        }
        filling_second = 1 - filling_second;
    }
}
