#pragma once

#include "conv3x3_stream.hpp"

// What happens between two convolutions, read from a finished conv3x3_stream
// result (acc) and written as a stream in the order the next conv3x3_stream
// with its own input parallelism P consumes it: P channels at a time, every
// pixel in raster order, one chunk's worth of channels per cycle. The caller
// partitions acc cyclic 2 in dims 1 and 2 (a 2x2 block comes from four
// different memories) and complete in dim 3 (so any P channels of one pixel
// can be read in the same cycle).

static inline float max2(float a, float b) {
    #pragma HLS INLINE
    return a > b ? a : b;
}

// ReLU, then 2x2 max-pool: H x W -> H/2 x W/2.
template<int H, int W, int CH, int P>
void emit_relu_pool(const float acc[H][W][CH], hls::stream<vec_t<P> >& out) {
    static_assert(CH % P == 0, "CH must be a multiple of P");
    for (int c = 0; c < CH / P; c++)
        for (int y = 0; y < H / 2; y++)
            for (int x = 0; x < W / 2; x++) {
                #pragma HLS PIPELINE II=1
                vec_t<P> v;
                for (int p = 0; p < P; p++) {
                    #pragma HLS UNROLL
                    const int ch = c * P + p;
                    float m = max2(max2(acc[2 * y][2 * x][ch], acc[2 * y][2 * x + 1][ch]),
                                   max2(acc[2 * y + 1][2 * x][ch], acc[2 * y + 1][2 * x + 1][ch]));
                    v.v[p] = max2(m, 0.0f);   // max then ReLU == ReLU then max
                }
                out.write(v);
            }
}

// ReLU, 2x2 max-pool, then 2x nearest upsample: H x W -> H x W, every 2x2
// block replaced by its maximum.
template<int H, int W, int CH, int P>
void emit_relu_pool_up(const float acc[H][W][CH], hls::stream<vec_t<P> >& out) {
    static_assert(CH % P == 0, "CH must be a multiple of P");
    for (int c = 0; c < CH / P; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                #pragma HLS PIPELINE II=1
                const int y0 = y & ~1, x0 = x & ~1;
                vec_t<P> v;
                for (int p = 0; p < P; p++) {
                    #pragma HLS UNROLL
                    const int ch = c * P + p;
                    float m = max2(max2(acc[y0][x0][ch], acc[y0][x0 + 1][ch]),
                                   max2(acc[y0 + 1][x0][ch], acc[y0 + 1][x0 + 1][ch]));
                    v.v[p] = max2(m, 0.0f);
                }
                out.write(v);
            }
}

// ReLU, then 2x nearest upsample: H x W -> 2H x 2W.
template<int H, int W, int CH, int P>
void emit_relu_up(const float acc[H][W][CH], hls::stream<vec_t<P> >& out) {
    static_assert(CH % P == 0, "CH must be a multiple of P");
    for (int c = 0; c < CH / P; c++)
        for (int y = 0; y < 2 * H; y++)
            for (int x = 0; x < 2 * W; x++) {
                #pragma HLS PIPELINE II=1
                vec_t<P> v;
                for (int p = 0; p < P; p++) {
                    #pragma HLS UNROLL
                    const int ch = c * P + p;
                    v.v[p] = max2(acc[y / 2][x / 2][ch], 0.0f);
                }
                out.write(v);
            }
}

// No activation: every pixel's CH values together, in raster order (HWC).
template<int H, int W, int CH>
void emit_pixels(const float acc[H][W][CH], hls::stream<vec_t<CH> >& out) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            #pragma HLS PIPELINE II=1
            vec_t<CH> v;
            for (int ch = 0; ch < CH; ch++) {
                #pragma HLS UNROLL
                v.v[ch] = acc[y][x][ch];
            }
            out.write(v);
        }
}
