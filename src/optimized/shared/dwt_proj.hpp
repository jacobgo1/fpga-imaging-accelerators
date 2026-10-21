#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// dwt_proj: WaveletCNN's 1-level 1D Daubechies-4 DWT along the IN_CH spectral
// bands of each pixel, fused with its 1x1 projection layer (CAT_CH -> MID_CH).
//
// In the PyTorch model, dwt_spectral zero-pads each pixel's 110-band spectrum
// by PAD = 6 on each side, runs stride-2 8-tap lowpass (h0) and highpass (h1)
// filters to make 58 low and 58 high coefficients, concatenates [x, low, high]
// into 226 channels, and multiplies by projection's 12 x 226 1x1 weights.
//
// Because x[2m], x[2m + 1], low[m] and high[m] all come from the same 8-band
// sliding window x_pad[2m .. 2m + 7] (where x_pad[2m + 6] = x[2m] and
// x_pad[2m + 7] = x[2m + 1]), the 226-channel vector never needs to be stored:
// each step m = 0 .. 57 computes low[m] and high[m] from the window and adds
// their contributions -- and x[2m], x[2m + 1]'s -- straight into the 12
// projection accumulators. M_PAR steps of m (2 * M_PAR input bands) run in
// parallel per clock cycle.
template<int PH, int PW, int IN_CH, int DWT_LEN, int MID_CH, int M_PAR,
         int X_SHIFT, int BIAS_SHIFT, int OUT_SHIFT,
         const wgt_t (&H0)[1][1][DWT_LEN], const wgt_t (&H1)[1][1][DWT_LEN],
         const wgt_t (&PROJ_W)[MID_CH][IN_CH + 2 * ((IN_CH + DWT_LEN - 1) / 2)][1][1],
         const wgt_t (&PROJ_B)[MID_CH],
         bool DWT_DSP, bool PROJ_DSP>
void dwt_proj(hls::stream<vec_t<act8_t, 2 * M_PAR> >& in,
              hls::stream<vec_t<act_t, MID_CH> >& out,
              int height, int width) {
    constexpr int SUB_CH = (IN_CH + DWT_LEN - 1) / 2;
    constexpr int PAD = (2 * (SUB_CH - 1) - IN_CH + DWT_LEN) / 2;
    constexpr int P = 2 * M_PAR;
    constexpr int STEPS = (SUB_CH + M_PAR - 1) / M_PAR;
    static_assert(PAD == DWT_LEN - 2, "stride-2 DWT window aligns x[2m] with tap PAD");

    wgt_t h0[DWT_LEN], h1[DWT_LEN];
    #pragma HLS ARRAY_PARTITION variable=h0 type=complete
    #pragma HLS ARRAY_PARTITION variable=h1 type=complete
    load_h: for (int k = 0; k < DWT_LEN; k++) {
        #pragma HLS PIPELINE II=1
        h0[k] = H0[0][0][k];
        h1[k] = H1[0][0][k];
    }

    wgt_t wx0[STEPS][M_PAR][MID_CH], wx1[STEPS][M_PAR][MID_CH];
    wgt_t wlo[STEPS][M_PAR][MID_CH], whi[STEPS][M_PAR][MID_CH];
    #pragma HLS ARRAY_PARTITION variable=wx0 type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wx0 type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wx1 type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wx1 type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wlo type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wlo type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=whi type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=whi type=complete dim=3
    acc_t bl[MID_CH];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete

    load_w: for (int s = 0; s < STEPS; s++) {
        for (int u = 0; u < M_PAR; u++) {
            for (int oc = 0; oc < MID_CH; oc++) {
                #pragma HLS PIPELINE II=1
                const int m = s * M_PAR + u;
                const int c0 = 2 * m < IN_CH ? 2 * m : 0;
                const int c1 = 2 * m + 1 < IN_CH ? 2 * m + 1 : 0;
                const int cl = m < SUB_CH ? IN_CH + m : 0;
                const int ch = m < SUB_CH ? IN_CH + SUB_CH + m : 0;
                wx0[s][u][oc] = 2 * m < IN_CH ? PROJ_W[oc][c0][0][0] : (wgt_t)0;
                wx1[s][u][oc] = 2 * m + 1 < IN_CH ? PROJ_W[oc][c1][0][0] : (wgt_t)0;
                wlo[s][u][oc] = m < SUB_CH ? PROJ_W[oc][cl][0][0] : (wgt_t)0;
                whi[s][u][oc] = m < SUB_CH ? PROJ_W[oc][ch][0][0] : (wgt_t)0;
            }
        }
    }
    load_b: for (int oc = 0; oc < MID_CH; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc] = (acc_t)PROJ_B[oc] * ((acc_t)1 << BIAS_SHIFT);
    }

    act8_t hist[PAD];
    #pragma HLS ARRAY_PARTITION variable=hist type=complete
    acc_t acc[MID_CH];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete

    const int n_pixels = patch_count<PH, PW>(height, width) * PH * PW;
    int s = 0;
    pixel_step: for (int t = 0; t < n_pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8192 max=19747840
        const vec_t<act8_t, P> v = in.read();
        act8_t w[PAD + P];
        #pragma HLS ARRAY_PARTITION variable=w type=complete
        for (int k = 0; k < PAD; k++) {
            #pragma HLS UNROLL
            w[k] = (s == 0) ? (act8_t)0 : hist[k];
        }
        for (int k = 0; k < P; k++) {
            #pragma HLS UNROLL
            w[PAD + k] = v.v[k];
        }
        for (int k = 0; k < PAD; k++) {
            #pragma HLS UNROLL
            hist[k] = w[P + k];
        }

        act_t lo[M_PAR], hi[M_PAR];
        #pragma HLS ARRAY_PARTITION variable=lo type=complete
        #pragma HLS ARRAY_PARTITION variable=hi type=complete
        for (int u = 0; u < M_PAR; u++) {
            #pragma HLS UNROLL
            act_t sum_lo = 0, sum_hi = 0;
            for (int k = 0; k < DWT_LEN; k++) {
                #pragma HLS UNROLL
                sum_lo += mul<DWT_DSP, act_t>(w[2 * u + k], h0[k]);
                sum_hi += mul<DWT_DSP, act_t>(w[2 * u + k], h1[k]);
            }
            lo[u] = sum_lo;
            hi[u] = sum_hi;
        }

        vec_t<act_t, MID_CH> out_v;
        for (int oc = 0; oc < MID_CH; oc++) {
            #pragma HLS UNROLL
            acc_t delta = 0;
            for (int u = 0; u < M_PAR; u++) {
                #pragma HLS UNROLL
                const acc_t px = ((acc_t)mul<PROJ_DSP, int16_t>(w[2 * u + PAD], wx0[s][u][oc]) +
                                  (acc_t)mul<PROJ_DSP, int16_t>(w[2 * u + PAD + 1], wx1[s][u][oc])) << X_SHIFT;
                const acc_t pd = mul<PROJ_DSP, acc_t>(lo[u], wlo[s][u][oc]) +
                                 mul<PROJ_DSP, acc_t>(hi[u], whi[s][u][oc]);
                delta += px + pd;
            }
            const acc_t total = (s == 0 ? bl[oc] : acc[oc]) + delta;
            acc[oc] = total;
            out_v.v[oc] = requant_act<OUT_SHIFT>(total);
        }

        if (s == STEPS - 1) {
            out.write(out_v);
            s = 0;
        } else {
            s++;
        }
    }
}
