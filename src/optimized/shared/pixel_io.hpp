#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"
#include "fixed_point.hpp"

// DDR in and out for the per-pixel (1D) networks: no patches, the image is one
// run of height x width pixels in memory order. The readers take turns, a block
// of pixels each. A network small enough to copy runs once per reader (a lane),
// on that reader's blocks only, and store_pixels_lanes puts the scores back in order.

// height x width in LUTs; the index arithmetic is 16-bit.
static inline uint32_t pixel_count(int height, int width) {
    #pragma HLS INLINE
    return mul<false, uint32_t>((uint16_t)height, (uint16_t)width);
}

// The pixels of one lane: blocks LANE, LANE + LANES, ... of BLOCK pixels (the
// image's last block may be shorter). pixel_lane<BLOCK, 1, 0> is the whole image.
template<int BLOCK, int LANES, int LANE>
struct pixel_lane {
    static uint32_t count(int height, int width) {
        #pragma HLS INLINE
        const uint32_t pixels = pixel_count(height, width);
        const uint32_t full_blocks = pixels / BLOCK, rest = pixels % BLOCK;
        const uint32_t mine = full_blocks > (uint32_t)LANE ? (full_blocks - LANE - 1) / LANES + 1 : 0;
        return mine * BLOCK + (rest != 0 && full_blocks % LANES == (uint32_t)LANE ? rest : 0);
    }
};

// DDR to stream: blocks PORT, PORT + READ_PORTS, ... of BLOCK pixels, one burst a
// block, requested BURSTS_AHEAD blocks ahead of the reading.
template<int WORDS_PER_PIXEL, int BLOCK, int READ_PORTS, int PORT, int BURSTS_AHEAD = 2, typename WORD_T>
void read_pixels_ahead(hls::burst_maxi<WORD_T> din, hls::stream<WORD_T>& out, int height, int width) {
    const uint32_t pixels = pixel_count(height, width);
    const uint32_t blocks = (pixels + BLOCK - 1) / BLOCK;
    uint32_t next_request = PORT;
    prime: for (int k = 0; k < BURSTS_AHEAD; k++) {
        if (next_request < blocks) {
            const uint32_t length = pixels - next_request * BLOCK < (uint32_t)BLOCK ? pixels - next_request * BLOCK
                                                                                    : (uint32_t)BLOCK;
            din.read_request(next_request * BLOCK * WORDS_PER_PIXEL, length * WORDS_PER_PIXEL);
            next_request += READ_PORTS;
        }
    }
    block: for (uint32_t b = PORT; b < blocks; b += READ_PORTS) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=20408
        if (next_request < blocks) {
            const uint32_t length = pixels - next_request * BLOCK < (uint32_t)BLOCK ? pixels - next_request * BLOCK
                                                                                    : (uint32_t)BLOCK;
            din.read_request(next_request * BLOCK * WORDS_PER_PIXEL, length * WORDS_PER_PIXEL);
            next_request += READ_PORTS;
        }
        const uint32_t length = pixels - b * BLOCK < (uint32_t)BLOCK ? pixels - b * BLOCK : (uint32_t)BLOCK;
        take: for (uint32_t k = 0; k < length * WORDS_PER_PIXEL; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=15 max=240
            out.write(din.read());
        }
    }
}

// One band's z-score in integers: (raw * scale + offset) >> ZSCORE_SHIFT, saturated.
template<int ZSCORE_SHIFT, int ZSCORE_MAX, bool DSP>
int16_t zscore12(uint16_t raw, int32_t scale, int64_t offset) {
    #pragma HLS INLINE
    const int64_t z = (mul<DSP, int64_t>(raw, scale) + offset) >> ZSCORE_SHIFT;
    return (int16_t)(z > ZSCORE_MAX ? ZSCORE_MAX : z < -ZSCORE_MAX ? -ZSCORE_MAX : z);
}

// A pixel's whole spectrum or feature map, moved as one wide word.
template<typename T, int LENGTH, int CHANNELS>
struct map_t {
    T v[LENGTH][CHANNELS];
};

// The readers' words to each pixel's z-scored spectrum, in pixel order, one word
// a cycle. The spectrum is shifted in a word at a time, so every index is a constant.
template<int WORDS_PER_PIXEL, int FIRST_KEPT_WORD, int BANDS_PER_WORD, int BLOCK, int READ_PORTS,
         int ZSCORE_SHIFT, int ZSCORE_MAX,
         const int32_t (&ZSCORE_SCALE)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         const int64_t (&ZSCORE_OFFSET)[(WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD],
         bool DSP, typename WORD_T>
void zscore_pixels(hls::stream<WORD_T> raw[READ_PORTS],
                   hls::stream<map_t<int16_t, (WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD, 1> >& out,
                   int height, int width) {
    constexpr int KEPT_WORDS = WORDS_PER_PIXEL - FIRST_KEPT_WORD;
    constexpr int KEPT_BANDS = KEPT_WORDS * BANDS_PER_WORD;
    int32_t scale[KEPT_WORDS][BANDS_PER_WORD];
    int64_t offset[KEPT_WORDS][BANDS_PER_WORD];
    #pragma HLS ARRAY_PARTITION variable=scale type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=offset type=complete dim=2
    load_z: for (int k = 0; k < KEPT_BANDS; k++) {
        #pragma HLS PIPELINE II=1
        scale[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_SCALE[k];
        offset[k / BANDS_PER_WORD][k % BANDS_PER_WORD] = ZSCORE_OFFSET[k];
    }

    const uint32_t pixels = pixel_count(height, width);
    map_t<int16_t, KEPT_BANDS, 1> spectrum;
    #pragma HLS ARRAY_PARTITION variable=spectrum.v type=complete dim=0
    int word_of_pixel = 0, pixel_of_block = 0, port = 0;
    word: for (uint32_t t = 0; t < pixels * WORDS_PER_PIXEL; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=15 max=9795240
        WORD_T word;
        for (int r = 0; r < READ_PORTS; r++) {
            #pragma HLS UNROLL
            if (port == r) word = raw[r].read();
        }
        if (word_of_pixel >= FIRST_KEPT_WORD) {
            const int kept_word = word_of_pixel - FIRST_KEPT_WORD;
            for (int b = 0; b < KEPT_BANDS - BANDS_PER_WORD; b++) {
                #pragma HLS UNROLL
                spectrum.v[b][0] = spectrum.v[b + BANDS_PER_WORD][0];
            }
            for (int b = 0; b < BANDS_PER_WORD; b++) {
                #pragma HLS UNROLL
                spectrum.v[KEPT_BANDS - BANDS_PER_WORD + b][0] = zscore12<ZSCORE_SHIFT, ZSCORE_MAX, DSP>(
                    word[b], scale[kept_word][b], offset[kept_word][b]);
            }
        }
        if (word_of_pixel == WORDS_PER_PIXEL - 1) {
            out.write(spectrum);
            word_of_pixel = 0;
            if (pixel_of_block == BLOCK - 1) { pixel_of_block = 0; port = port == READ_PORTS - 1 ? 0 : port + 1; }
            else pixel_of_block++;
        } else {
            word_of_pixel++;
        }
    }
}

// Each pixel's scores and its class (the first highest score) into DDR in pixel
// order, bursts of BURST pixels, acknowledgements collected BURSTS_IN_FLIGHT
// bursts later; then a word on finished, for count_cycles.
template<int CLASSES, typename WORD_T>
WORD_T pixel_scores_and_class(const vec_t<acc_t, CLASSES>& scores) {
    #pragma HLS INLINE
    WORD_T word;
    int best = 0;
    for (int c = 0; c < CLASSES; c++) {
        #pragma HLS UNROLL
        word[c] = scores.v[c];
        if (scores.v[c] > scores.v[best]) best = c;
    }
    word[CLASSES] = best;
    return word;
}

template<int CLASSES, int BURST, int BURSTS_IN_FLIGHT, typename WORD_T>
void store_pixels(hls::stream<vec_t<acc_t, CLASSES> >& in, hls::burst_maxi<WORD_T> dout, int height, int width,
                  hls::stream<bool>& finished) {
    const uint32_t pixels = pixel_count(height, width);
    int unacknowledged = 0;
    burst: for (uint32_t first = 0; first < pixels; first += BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=10204
        const uint32_t length = pixels - first < (uint32_t)BURST ? pixels - first : (uint32_t)BURST;
        if (unacknowledged == BURSTS_IN_FLIGHT) dout.write_response();
        else unacknowledged++;
        dout.write_request(first, length);
        put: for (uint32_t k = 0; k < length; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            dout.write(pixel_scores_and_class<CLASSES, WORD_T>(in.read()));
        }
    }
    drain: for (int k = 0; k < unacknowledged; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        dout.write_response();
    }
    finished.write(true);
}

// store_pixels for a network in LANES lanes: the lanes are taken in turn, BLOCK
// pixels each, the order their readers read the image in.
template<int CLASSES, int LANES, int BLOCK, int BURST, int BURSTS_IN_FLIGHT, typename WORD_T>
void store_pixels_lanes(hls::stream<vec_t<acc_t, CLASSES> > in[LANES], hls::burst_maxi<WORD_T> dout,
                        int height, int width, hls::stream<bool>& finished) {
    const uint32_t pixels = pixel_count(height, width);
    int unacknowledged = 0;
    int pixel_of_block = 0, lane = 0;
    burst: for (uint32_t first = 0; first < pixels; first += BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=10204
        const uint32_t length = pixels - first < (uint32_t)BURST ? pixels - first : (uint32_t)BURST;
        if (unacknowledged == BURSTS_IN_FLIGHT) dout.write_response();
        else unacknowledged++;
        dout.write_request(first, length);
        put: for (uint32_t k = 0; k < length; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            vec_t<acc_t, CLASSES> scores;
            for (int r = 0; r < LANES; r++) {
                #pragma HLS UNROLL
                if (lane == r) scores = in[r].read();
            }
            dout.write(pixel_scores_and_class<CLASSES, WORD_T>(scores));
            if (pixel_of_block == BLOCK - 1) { pixel_of_block = 0; lane = lane == LANES - 1 ? 0 : lane + 1; }
            else pixel_of_block++;
        }
    }
    drain: for (int k = 0; k < unacknowledged; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        dout.write_response();
    }
    finished.write(true);
}
