#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"
#include "fixed_point.hpp"

// DDR in and out for the per-pixel (1D) kernels. A 1D network classifies each
// pixel from its own spectrum, so there are no patches, no halos and no edge
// padding: the image is one run of height x width pixels, [pixel][WORDS] words,
// read and written in memory order.
//
//   read_pixels_ahead<0> ─┐  blocks 0, 2, 4, ... (HP0)
//   read_pixels_ahead<1> ─┴─► zscore_pixels ─► the network ─► store_pixels (HP2)
//                             blocks 1, 3, 5, ... (HP1)
//
// A network small enough to copy can instead run once per reader, each copy (a
// lane) on its reader's blocks only, and store_pixels_lanes puts the lanes'
// scores back in pixel order:
//
//   read_pixels_ahead<0> ─► the network, lane 0 ─┐
//   read_pixels_ahead<1> ─► the network, lane 1 ─┼─► store_pixels_lanes
//   ...                                          ┘

// pixels = height x width, in LUTs (the index arithmetic is 16-bit, as in patch_grid.hpp).
static inline uint32_t pixel_count(int height, int width) {
    #pragma HLS INLINE
    return mul<false, uint32_t>((uint16_t)height, (uint16_t)width);
}

// pixel_lane: the pixels of one lane, the blocks LANE, LANE + LANES, LANE + 2
// LANES, ... of BLOCK pixels (the last block of the image maybe shorter), what
// read_pixels_ahead<..., LANES, LANE> reads. A lane's processes take it as a
// template argument and ask it how many pixels they get; pixel_lane<BLOCK, 1,
// 0> is the whole image.
template<int BLOCK, int LANES, int LANE>
struct pixel_lane {
    static_assert(LANE < LANES, "a lane of LANES");
    static uint32_t count(int height, int width) {
        #pragma HLS INLINE
        const uint32_t pixels = pixel_count(height, width);
        const uint32_t full = pixels / BLOCK, rest = pixels % BLOCK;    // whole blocks, and the short one
        const uint32_t mine = full > (uint32_t)LANE ? (full - LANE - 1) / LANES + 1 : 0;
        return mine * BLOCK + (rest != 0 && full % LANES == (uint32_t)LANE ? rest : 0);
    }
};

// read_pixels_ahead: DDR -> stream, the words of the pixels of blocks PORT,
// PORT + PORTS, PORT + 2 PORTS, ... (BLOCK pixels each, the last one maybe
// shorter): PORTS of these on their own m_axi ports read the image between
// them. A block is one burst of BLOCK x WORDS consecutive words (at most 256,
// AXI's limit), and the bursts are requested AHEAD blocks ahead of the reading
// (as read_rows_ahead does), so the DDR's latency is paid once per start. At
// most AHEAD + 1 requests are outstanding: keep that within the port's
// num_read_outstanding.
template<int WORDS, int BLOCK, int PORTS, int PORT, int AHEAD = 2, typename WORD_T>
void read_pixels_ahead(hls::burst_maxi<WORD_T> din, hls::stream<WORD_T>& out, int height, int width) {
    static_assert(BLOCK * WORDS <= 256, "a burst is at most 256 beats");
    static_assert(AHEAD >= 1 && PORT < PORTS, "at least one burst ahead; a port of PORTS");
    const uint32_t pixels = pixel_count(height, width);
    const uint32_t blocks = (pixels + BLOCK - 1) / BLOCK;
    uint32_t next = PORT;                   // the next block to request
    prime: for (int k = 0; k < AHEAD; k++) {
        if (next < blocks) {
            const uint32_t n = pixels - next * BLOCK < (uint32_t)BLOCK ? pixels - next * BLOCK : (uint32_t)BLOCK;
            din.read_request(next * BLOCK * WORDS, n * WORDS);
            next += PORTS;
        }
    }
    block: for (uint32_t b = PORT; b < blocks; b += PORTS) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=20408
        // The block AHEAD from this one, before this one's words.
        if (next < blocks) {
            const uint32_t n = pixels - next * BLOCK < (uint32_t)BLOCK ? pixels - next * BLOCK : (uint32_t)BLOCK;
            din.read_request(next * BLOCK * WORDS, n * WORDS);
            next += PORTS;
        }
        const uint32_t n = pixels - b * BLOCK < (uint32_t)BLOCK ? pixels - b * BLOCK : (uint32_t)BLOCK;
        take: for (uint32_t k = 0; k < n * WORDS; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=15 max=240
            out.write(din.read());
        }
    }
}

// The z-score of one band, in integers: (dn * a + b) >> FRAC, saturated at
// +-IN_MAX.
template<int FRAC, int IN_MAX, bool DSP>
int16_t zscore12(uint16_t dn, int32_t a, int64_t b) {
    #pragma HLS INLINE
    const int64_t s = (mul<DSP, int64_t>(dn, a) + b) >> FRAC;
    return (int16_t)(s > IN_MAX ? IN_MAX : s < -IN_MAX ? -IN_MAX : s);
}

// A pixel's whole spectrum (or a layer's whole feature map): L positions x C
// channels, moved as one wide word per pixel.
template<typename T, int L, int C>
struct map_t {
    T v[L][C];
};

// zscore_pixels: the readers' words -> each pixel's z-scored spectrum, in
// pixel order (blocks alternate between the PORTS readers). A pixel is WORDS
// words of LANES raw bands; words FIRST .. WORDS - 1 are kept, so kept band k
// is word FIRST + k / LANES, lane k % LANES: x = (dn * ZA[k] + ZB[k]) >> FRAC,
// saturated at +-IN_MAX. One word per cycle: WORDS cycles a pixel. The spectrum
// is built in a shift register (each word shifted in at the end, so every index
// is a constant: no multiplexers) and leaves as one map_t per pixel.
// DSP: its LANES z-score multipliers in DSPs, or in LUTs.
template<int WORDS, int FIRST, int LANES, int BLOCK, int PORTS, int FRAC, int IN_MAX,
         const int32_t (&ZA)[(WORDS - FIRST) * LANES], const int64_t (&ZB)[(WORDS - FIRST) * LANES],
         bool DSP, typename WORD_T>
void zscore_pixels(hls::stream<WORD_T> raw[PORTS], hls::stream<map_t<int16_t, (WORDS - FIRST) * LANES, 1> >& out,
                   int height, int width) {
    constexpr int KEPT = WORDS - FIRST;           // kept words per pixel
    constexpr int BANDS = KEPT * LANES;
    // The z-score constants, one small ROM per lane, addressed by the kept word.
    int32_t za[KEPT][LANES];
    int64_t zb[KEPT][LANES];
    #pragma HLS ARRAY_PARTITION variable=za type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=zb type=complete dim=2
    load_z: for (int k = 0; k < BANDS; k++) {
        #pragma HLS PIPELINE II=1
        za[k / LANES][k % LANES] = ZA[k];
        zb[k / LANES][k % LANES] = ZB[k];
    }

    const uint32_t pixels = pixel_count(height, width);
    map_t<int16_t, BANDS, 1> x;                   // the spectrum, shifted in a word at a time
    #pragma HLS ARRAY_PARTITION variable=x.v type=complete dim=0
    int w = 0, in_block = 0, port = 0;            // word of the pixel, pixel of the block, its reader
    word: for (uint32_t t = 0; t < pixels * WORDS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=15 max=9795240
        WORD_T word;
        for (int r = 0; r < PORTS; r++) {
            #pragma HLS UNROLL
            if (port == r) word = raw[r].read();
        }
        if (w >= FIRST) {
            const int kw = w - FIRST;
            for (int b = 0; b < BANDS - LANES; b++) {
                #pragma HLS UNROLL
                x.v[b][0] = x.v[b + LANES][0];
            }
            for (int p = 0; p < LANES; p++) {
                #pragma HLS UNROLL
                x.v[BANDS - LANES + p][0] = zscore12<FRAC, IN_MAX, DSP>(word[p], za[kw][p], zb[kw][p]);
            }
        }
        if (w == WORDS - 1) {
            out.write(x);
            w = 0;
            if (in_block == BLOCK - 1) { in_block = 0; port = port == PORTS - 1 ? 0 : port + 1; }
            else in_block++;
        } else {
            w++;
        }
    }
}

// store_pixels: one word per pixel (the CLASSES scores, then the class: the
// first highest score, like numpy's argmax) into DDR in pixel order, bursts of
// BURST words. Each burst's acknowledgement is collected IN_FLIGHT bursts later
// (as store_scores does), the last ones at the end; then one word on finished,
// once every word is in memory, for count_cycles.
template<int CLASSES, int BURST, int IN_FLIGHT, typename OUT_T>
void store_pixels(hls::stream<vec_t<acc_t, CLASSES> >& in, hls::burst_maxi<OUT_T> dout, int height, int width,
                  hls::stream<bool>& finished) {
    static_assert(BURST <= 256, "a burst is at most 256 beats");
    const uint32_t pixels = pixel_count(height, width);
    int pending = 0;                              // bursts not yet acknowledged
    burst: for (uint32_t p0 = 0; p0 < pixels; p0 += BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=10204
        const uint32_t n = pixels - p0 < (uint32_t)BURST ? pixels - p0 : (uint32_t)BURST;
        if (pending == IN_FLIGHT) dout.write_response();
        else pending++;
        dout.write_request(p0, n);
        put: for (uint32_t k = 0; k < n; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            const vec_t<acc_t, CLASSES> s = in.read();
            OUT_T o;
            int best = 0;
            for (int c = 0; c < CLASSES; c++) {
                #pragma HLS UNROLL
                o[c] = s.v[c];
                if (s.v[c] > s.v[best]) best = c;
            }
            o[CLASSES] = best;
            dout.write(o);
        }
    }
    drain: for (int k = 0; k < pending; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        dout.write_response();
    }
    finished.write(true);
}

// store_pixels_lanes: store_pixels for a network run in LANES lanes (pixel_lane):
// the same words, bursts and acknowledgements, but pixel p's scores come from
// lane (p / BLOCK) % LANES, so the lanes are taken in turn, BLOCK pixels each,
// the order their readers read the image in.
template<int CLASSES, int LANES, int BLOCK, int BURST, int IN_FLIGHT, typename OUT_T>
void store_pixels_lanes(hls::stream<vec_t<acc_t, CLASSES> > in[LANES], hls::burst_maxi<OUT_T> dout,
                        int height, int width, hls::stream<bool>& finished) {
    static_assert(BURST <= 256, "a burst is at most 256 beats");
    const uint32_t pixels = pixel_count(height, width);
    int pending = 0;                              // bursts not yet acknowledged
    int in_block = 0, lane = 0;                   // pixel of the block, its lane
    burst: for (uint32_t p0 = 0; p0 < pixels; p0 += BURST) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=10204
        const uint32_t n = pixels - p0 < (uint32_t)BURST ? pixels - p0 : (uint32_t)BURST;
        if (pending == IN_FLIGHT) dout.write_response();
        else pending++;
        dout.write_request(p0, n);
        put: for (uint32_t k = 0; k < n; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            vec_t<acc_t, CLASSES> s;
            for (int r = 0; r < LANES; r++) {
                #pragma HLS UNROLL
                if (lane == r) s = in[r].read();
            }
            OUT_T o;
            int best = 0;
            for (int c = 0; c < CLASSES; c++) {
                #pragma HLS UNROLL
                o[c] = s.v[c];
                if (s.v[c] > s.v[best]) best = c;
            }
            o[CLASSES] = best;
            dout.write(o);
            if (in_block == BLOCK - 1) { in_block = 0; lane = lane == LANES - 1 ? 0 : lane + 1; }
            else in_block++;
        }
    }
    drain: for (int k = 0; k < pending; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        dout.write_response();
    }
    finished.write(true);
}
