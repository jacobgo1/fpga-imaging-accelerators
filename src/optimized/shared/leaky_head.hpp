#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// leaky_head: WaveletCNN's activation after conv_block (requant + LeakyReLU)
// fused with seg_head (1x1 Conv2d(MID_CH -> OUT_CH) + folded BatchNorm +
// LeakyReLU), one pixel per clock cycle (II = 1, MID_CH x OUT_CH = 36
// multipliers).
//
// Out: for each pixel in raster order, the OUT_CH scores at OUT_FRAC fraction
// bits, ready for store_scores<PH, PW, 0>.
template<int PH, int PW, int MID_CH, int OUT_CH,
         int CONV_SHIFT, int HEAD_BIAS_SHIFT, int HEAD_SHIFT,
         const wgt_t (&HEAD_W)[OUT_CH][MID_CH][1][1], const wgt_t (&HEAD_B)[OUT_CH],
         bool HEAD_DSP>
void leaky_head(hls::stream<vec_t<acc_t, MID_CH> >& in,
                hls::stream<vec_t<acc_t, OUT_CH> >& out,
                int height, int width) {
    wgt_t wl[OUT_CH][MID_CH];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=0
    acc_t bl[OUT_CH];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete

    load_w: for (int oc = 0; oc < OUT_CH; oc++) {
        #pragma HLS PIPELINE II=1
        bl[oc] = (acc_t)HEAD_B[oc] * ((acc_t)1 << HEAD_BIAS_SHIFT);
        for (int ic = 0; ic < MID_CH; ic++) {
            #pragma HLS UNROLL
            wl[oc][ic] = HEAD_W[oc][ic][0][0];
        }
    }

    const int total = patch_count<PH, PW>(height, width) * PH * PW;
    pixel: for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=680960
        const vec_t<acc_t, MID_CH> v = in.read();
        act_t act1[MID_CH];
        #pragma HLS ARRAY_PARTITION variable=act1 type=complete
        for (int ic = 0; ic < MID_CH; ic++) {
            #pragma HLS UNROLL
            act1[ic] = requant_leaky_relu<CONV_SHIFT>(v.v[ic]);
        }

        vec_t<acc_t, OUT_CH> o;
        for (int oc = 0; oc < OUT_CH; oc++) {
            #pragma HLS UNROLL
            acc_t sum = bl[oc];
            for (int ic = 0; ic < MID_CH; ic++) {
                #pragma HLS UNROLL
                sum += mul<HEAD_DSP, acc_t>(act1[ic], wl[oc][ic]);
            }
            o.v[oc] = leaky_relu<>(round_shift<HEAD_SHIFT>(sum));
        }
        out.write(o);
    }
}
