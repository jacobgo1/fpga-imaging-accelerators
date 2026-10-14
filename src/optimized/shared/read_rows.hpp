#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"
#include "patch_grid.hpp"

// read_rows: DDR -> stream, the raw words of every PORTS-th row of each patch.
// din is the image, [height][width][WORDS] words (a word: some bands of a
// pixel, one bus beat). For each patch, rows PORT, PORT + PORTS, ... : PW
// pixels x WORDS words each, in memory order, one burst per row. PORTS of
// these, each on its own m_axi port, together read every row.
// Past the image's bottom the last row is read again; past its right edge the
// last pixel is repeated, so every patch is whole.
template<int PH, int PW, int WORDS, int PORTS, int PORT, typename WORD_T>
void read_rows(const WORD_T *din, hls::stream<WORD_T>& out, int height, int width) {
    static_assert(PH % PORTS == 0, "every port reads as many rows of a patch");
    constexpr int ROWS = PH / PORTS;
    const int rows = patch_rows<PH>(height), cols = patch_cols<PW>(width);
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int i = 0; i < ROWS; i++) {
                const int y = pr * PH + PORTS * i + PORT;
                const int x0 = pc * PW;
                const int n = width - x0 < PW ? width - x0 : PW;   // pixels inside the image
                const uint32_t row = mul<false, uint32_t>((uint16_t)(y < height ? y : height - 1),
                                                          (uint16_t)width);   // in LUTs
                const WORD_T *src = din + (row + x0) * WORDS;
                WORD_T last[WORDS];                                // the last pixel read
                #pragma HLS ARRAY_PARTITION variable=last type=complete
                int w = 0;
                take: for (int k = 0; k < n * WORDS; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=15 max=480       // 15 words per pixel, 32 pixels
                    const WORD_T v = src[k];
                    out.write(v);
                    last[w] = v;
                    w = (w == WORDS - 1) ? 0 : w + 1;
                }
                edge: for (int k = 0; k < (PW - n) * WORDS; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=465
                    out.write(last[w]);
                    w = (w == WORDS - 1) ? 0 : w + 1;
                }
            }
        }
    }
}

// read_rows_ahead: read_rows on an hls::burst_maxi port, with the bursts
// spelled out so that the next row is requested while this one's words still
// come in. read_rows asks for a row only when it starts on it and then waits
// the memory's whole latency (tens of cycles through the ZCU104's PS) before
// the first word: ROWS times per patch, and with four ports reading at once the
// patch waits for the slowest. Here each row is asked for in bursts of SEG
// pixels (SEG * WORDS words: AXI allows at most 256 beats a burst), and the
// requests run AHEAD bursts ahead of the reading -- with the default 2 and
// SEG 16, a whole 32-pixel row. At most AHEAD + 1 requests are outstanding:
// keep that within the port's num_read_outstanding. Same words, same order,
// same edge handling as read_rows.

// The burst at a request cursor: where in din it starts, and how many words.
template<int PH, int PW, int WORDS, int PORTS, int PORT, int SEG>
void burst_at(int pr, int pc, int i, int s, int height, int width, uint32_t& offset, int& len) {
    #pragma HLS INLINE
    const int y = pr * PH + PORTS * i + PORT;
    const int x0 = pc * PW;
    const int n = width - x0 < PW ? width - x0 : PW;            // pixels inside the image
    const int px = n - s * SEG < SEG ? n - s * SEG : SEG;        // pixels in this burst
    const uint32_t row = mul<false, uint32_t>((uint16_t)(y < height ? y : height - 1),
                                              (uint16_t)width);   // in LUTs
    offset = (row + x0 + s * SEG) * WORDS;
    len = px * WORDS;
}

// The request cursor's next burst, in reading order; false past the last one.
template<int PW, int ROWS, int SEG>
bool next_burst(int& pr, int& pc, int& i, int& s, int rows, int cols, int width) {
    #pragma HLS INLINE
    const int n = width - pc * PW < PW ? width - pc * PW : PW;
    if (++s < (n + SEG - 1) / SEG) return true;
    s = 0;
    if (++i < ROWS) return true;
    i = 0;
    if (++pc < cols) return true;
    pc = 0;
    return ++pr < rows;
}

template<int PH, int PW, int WORDS, int PORTS, int PORT, int SEG = 16, int AHEAD = 2, typename WORD_T>
void read_rows_ahead(hls::burst_maxi<WORD_T> din, hls::stream<WORD_T>& out, int height, int width) {
    static_assert(PH % PORTS == 0, "every port reads as many rows of a patch");
    static_assert(SEG * WORDS <= 256, "a burst is at most 256 beats");
    static_assert(AHEAD >= 1, "at least one burst requested ahead");
    constexpr int ROWS = PH / PORTS;
    const int rows = patch_rows<PH>(height), cols = patch_cols<PW>(width);

    // The request cursor: patch row, patch column, line of the port, burst of the line.
    int qr = 0, qc = 0, qi = 0, qs = 0;
    bool more = rows > 0 && cols > 0;
    uint32_t offset;
    int len;
    prime: for (int k = 0; k < AHEAD; k++) {
        if (more) {
            burst_at<PH, PW, WORDS, PORTS, PORT, SEG>(qr, qc, qi, qs, height, width, offset, len);
            din.read_request(offset, len);
            more = next_burst<PW, ROWS, SEG>(qr, qc, qi, qs, rows, cols, width);
        }
    }
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int i = 0; i < ROWS; i++) {
                const int x0 = pc * PW;
                const int n = width - x0 < PW ? width - x0 : PW;   // pixels inside the image
                WORD_T last[WORDS];                                // the last pixel read
                #pragma HLS ARRAY_PARTITION variable=last type=complete
                int w = 0;
                burst: for (int s = 0; s * SEG < n; s++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=2
                    // The burst AHEAD from this one, before this one's words.
                    if (more) {
                        burst_at<PH, PW, WORDS, PORTS, PORT, SEG>(qr, qc, qi, qs, height, width, offset, len);
                        din.read_request(offset, len);
                        more = next_burst<PW, ROWS, SEG>(qr, qc, qi, qs, rows, cols, width);
                    }
                    const int words = (n - s * SEG < SEG ? n - s * SEG : SEG) * WORDS;
                    take: for (int k = 0; k < words; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=15 max=240     // 15 words per pixel, 16 pixels
                        const WORD_T v = din.read();
                        out.write(v);
                        last[w] = v;
                        w = (w == WORDS - 1) ? 0 : w + 1;
                    }
                }
                edge: for (int k = 0; k < (PW - n) * WORDS; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=465
                    out.write(last[w]);
                    w = (w == WORDS - 1) ? 0 : w + 1;
                }
            }
        }
    }
}
