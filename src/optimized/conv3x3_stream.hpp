#pragma once

#include "../common/hls_stream_shim.hpp"

// Optimized building blocks for streaming CNN kernels (float activations).
// The golden/ templates say *what* is computed; these say *how fast*: they read
// their input from an hls::stream in the order they consume it, keep only two
// lines of it on chip, and do many multiply-adds per clock cycle.

// P values that travel together through a stream (P input channels of one pixel).
template<int P>
struct vec_t {
    float v[P];
};

// Copy one conv layer's weights from ROM into a local array, with the
// power-of-two scales folded in (q * 2^-frac is exact in float, so this changes
// no result). The caller declares wl/bl and partitions them as conv3x3_stream
// needs: wl complete in dims 1, 3, 4 and cyclic by P in dim 2; bl complete.
template<int OUT_CH, int IN_CH>
void load_conv_weights(const float w[OUT_CH][IN_CH][3][3], float w_scale,
                       const float b[OUT_CH], float b_scale,
                       float wl[OUT_CH][IN_CH][3][3], float bl[OUT_CH]) {
    load_w: for (int oc = 0; oc < OUT_CH; oc++)
        for (int ic = 0; ic < IN_CH; ic++)
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++) {
                    #pragma HLS PIPELINE II=1
                    wl[oc][ic][ki][kj] = w[oc][ic][ki][kj] * w_scale;
                }
    load_b: for (int oc = 0; oc < OUT_CH; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc] = b[oc] * b_scale;
    }
}

// Same-padded (zero) 3x3 convolution of one H x W image, streamed.
//
// Input order: P channels at a time, plane by plane -- for each chunk c of P
// channels, every pixel in raster order (the stream carries IN_CH/P * H * W
// vectors). That is the order the computation uses them in, so no input image is
// stored: a 2-line buffer plus a 3x3 window hold all a chunk needs.
//
// Output: acc[y][x][oc] = bias + the full sum over all IN_CH channels, built up
// one chunk at a time. The caller partitions acc complete in dim 3 (all output
// channels written in the same cycle).
//
// Per loop iteration (II cycles): OUT_CH x P x 9 multiply-adds, fully unrolled.
// II > 1 makes HLS share the multipliers and adders between II cycles, for the
// small layers that have time to spare.
template<int H, int W, int IN_CH, int OUT_CH, int P, int II>
void conv3x3_stream(hls::stream<vec_t<P> >& in,
                    const float wl[OUT_CH][IN_CH][3][3], const float bl[OUT_CH],
                    float acc[H][W][OUT_CH]) {
    static_assert(IN_CH % P == 0, "IN_CH must be a multiple of P");

    float lb[2][W][P];       // the two previous input lines of this chunk
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lb type=complete dim=3
    float win[3][3][P];      // the 3x3 window, in registers
    #pragma HLS ARRAY_PARTITION variable=win type=complete dim=0

    // One extra row and column at the end: the window is centred one pixel
    // behind the input, so the last row and column need a step of zero padding.
    // The three loops are flattened into one pipeline: no flush between chunks.
    chunk: for (int c = 0; c < IN_CH / P; c++) {
        row: for (int r = 0; r <= H; r++) {
            col: for (int s = 0; s <= W; s++) {
                #pragma HLS PIPELINE II=II
                // Each acc pixel is written once per chunk; the next chunk's
                // read of it is (H+1)(W+1) iterations later.
                #pragma HLS DEPENDENCE variable=acc type=inter false
                #pragma HLS DEPENDENCE variable=lb type=inter false
                const bool inside = r < H && s < W;
                vec_t<P> x;
                if (inside) x = in.read();

                // New window column: input rows r-2, r-1, r at column s
                // (zero above the image, below it, and right of it).
                float column[3][P];
                #pragma HLS ARRAY_PARTITION variable=column type=complete dim=0
                for (int p = 0; p < P; p++) {
                    #pragma HLS UNROLL
                    column[0][p] = (r >= 2 && s < W) ? lb[0][s][p] : 0.0f;
                    column[1][p] = (r >= 1 && s < W) ? lb[1][s][p] : 0.0f;
                    column[2][p] = inside ? x.v[p] : 0.0f;
                    if (s < W) {
                        lb[0][s][p] = lb[1][s][p];
                        lb[1][s][p] = column[2][p];
                    }
                }
                for (int ki = 0; ki < 3; ki++) {
                    #pragma HLS UNROLL
                    for (int p = 0; p < P; p++) {
                        #pragma HLS UNROLL
                        win[ki][0][p] = win[ki][1][p];
                        win[ki][1][p] = win[ki][2][p];
                        win[ki][2][p] = column[ki][p];
                    }
                }

                // The window is now centred on (r-1, s-1).
                if (r >= 1 && s >= 1) {
                    const int y = r - 1, xo = s - 1;
                    for (int oc = 0; oc < OUT_CH; oc++) {
                        #pragma HLS UNROLL
                        float sum = 0.0f;
                        for (int p = 0; p < P; p++) {
                            #pragma HLS UNROLL
                            for (int ki = 0; ki < 3; ki++) {
                                #pragma HLS UNROLL
                                for (int kj = 0; kj < 3; kj++) {
                                    #pragma HLS UNROLL
                                    sum += win[ki][kj][p] * wl[oc][c * P + p][ki][kj];
                                }
                            }
                        }
                        const float before = (c == 0) ? bl[oc] : acc[y][xo][oc];
                        acc[y][xo][oc] = before + sum;
                    }
                }
            }
        }
    }
}
