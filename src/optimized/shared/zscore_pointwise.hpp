#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "pixel_io.hpp"
#include "conv1d_engine.hpp"

// A per-pixel network's first layer, a 1x1 conv on the z-scored bands, done on
// the raw words as they come from the reader: one word a cycle, its bands
// z-scored (as zscore_pixels) and multiplied into all output channels' sums.
// After the pixel's last word: ReLU, SHIFT bits rounded off, saturated at
// ACT_MAX. Words before FIRST_KEPT_WORD are dropped; kept bands past
// IN_CHANNELS get weight 0. The 12-bit z-scores are too wide to share a DSP.
// Each (band lane, output channel) has its own memory: the run-time choice is
// the address, never the bank. LANE: which pixels (pixel_lane).
template<int WORDS_PER_PIXEL, int FIRST_KEPT_WORD, int BANDS_PER_WORD, int ZSCORE_SHIFT, int ZSCORE_MAX,
         const int32_t (&ZSCORE_SCALE)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         const int64_t (&ZSCORE_OFFSET)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         int IN_CHANNELS, int OUT_CHANNELS, int BIAS_SHIFT, int SHIFT, int ACT_MAX,
         const wgt_t (&WEIGHT)[OUT_CHANNELS][IN_CHANNELS][1][1], const wgt_t (&BIAS)[OUT_CHANNELS],
         bool ZSCORE_DSP, bool DSP, typename LANE, typename WORD_T>
void zscore_pointwise(hls::stream<WORD_T>& raw, hls::stream<vec_t<uact_t, OUT_CHANNELS> >& out,
                      int height, int width) {
    constexpr int KEPT_WORDS = WORDS_PER_PIXEL - FIRST_KEPT_WORD;
    constexpr int KEPT_BANDS = KEPT_WORDS * BANDS_PER_WORD;
    int32_t scale[KEPT_WORDS][BANDS_PER_WORD];
    int64_t offset[KEPT_WORDS][BANDS_PER_WORD];
    wgt_t weight[KEPT_WORDS][BANDS_PER_WORD][OUT_CHANNELS];
    acc_t bias[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=scale type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=offset type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=bias type=complete
    load_z: for (int k = 0; k < KEPT_BANDS; k++) {
        scale[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_SCALE[k];
        offset[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_OFFSET[k];
        for (int o = 0; o < OUT_CHANNELS; o++) {
            #pragma HLS PIPELINE II=1
            const int source = k < IN_CHANNELS ? k : IN_CHANNELS - 1;   // never read past WEIGHT
            weight[k / BANDS_PER_WORD][k % BANDS_PER_WORD][o] = k < IN_CHANNELS ? WEIGHT[o][source][0][0] : (wgt_t)0;
        }
    }
    load_b: for (int o = 0; o < OUT_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o] = (acc_t)BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }

    const uint32_t pixels = LANE::count(height, width);
    acc_t sums[OUT_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete
    int word_of_pixel = 0;
    word: for (uint32_t t = 0; t < pixels * WORDS_PER_PIXEL; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=15 max=9795240
        const WORD_T word = raw.read();
        vec_t<uact_t, OUT_CHANNELS> activations;
        if (word_of_pixel >= FIRST_KEPT_WORD) {
            const int kept_word = word_of_pixel - FIRST_KEPT_WORD;
            int16_t zscores[BANDS_PER_WORD];
            #pragma HLS ARRAY_PARTITION variable=zscores type=complete
            for (int b = 0; b < BANDS_PER_WORD; b++) {
                #pragma HLS UNROLL
                zscores[b] = zscore12<ZSCORE_SHIFT, ZSCORE_MAX, ZSCORE_DSP>(
                    word[b], scale[kept_word][b], offset[kept_word][b]);
            }
            for (int o = 0; o < OUT_CHANNELS; o++) {
                #pragma HLS UNROLL
                acc_t part = 0;
                for (int b = 0; b < BANDS_PER_WORD; b++) {
                    #pragma HLS UNROLL
                    part += mul<DSP, acc_t>(zscores[b], weight[kept_word][b][o]);
                }
                const acc_t total = (word_of_pixel == FIRST_KEPT_WORD ? bias[o] : sums[o]) + part;
                sums[o] = total;
                activations.v[o] = requant_relu8<SHIFT, ACT_MAX>(total);
            }
        }
        if (word_of_pixel == WORDS_PER_PIXEL - 1) {
            out.write(activations);
            word_of_pixel = 0;
        } else {
            word_of_pixel++;
        }
    }
}
