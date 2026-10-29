#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"
#include "fixed_point.hpp"
#include "patch_grid.hpp"

// WaveletCNN's db4 wavelet transform along each pixel's bands, fused with the
// 1x1 projection after it. Bands 2m and 2m + 1, low[m] and high[m] all come from
// the same 8-band window, so the 226 channels are never stored: each step adds
// their contributions straight into the projection's sums. STEPS_PER_CYCLE steps
// of m (2 x STEPS_PER_CYCLE bands) a cycle.
template<int PATCH_HEIGHT, int PATCH_WIDTH, int BANDS, int TAPS, int MID_CHANNELS, int STEPS_PER_CYCLE,
         int BAND_SHIFT, int BIAS_SHIFT, int OUT_SHIFT,
         const wgt_t (&LOWPASS)[1][1][TAPS], const wgt_t (&HIGHPASS)[1][1][TAPS],
         const wgt_t (&PROJECTION_WEIGHT)[MID_CHANNELS][BANDS + 2 * ((BANDS + TAPS - 1) / 2)][1][1],
         const wgt_t (&PROJECTION_BIAS)[MID_CHANNELS],
         bool DWT_DSP, bool PROJECTION_DSP>
void dwt_proj(hls::stream<vec_t<act8_t, 2 * STEPS_PER_CYCLE> >& in, hls::stream<vec_t<act_t, MID_CHANNELS> >& out,
              int height, int width) {
    constexpr int SUBBAND = (BANDS + TAPS - 1) / 2;
    constexpr int PAD = (2 * (SUBBAND - 1) - BANDS + TAPS) / 2;   // zero bands before the first
    constexpr int BANDS_PER_CYCLE = 2 * STEPS_PER_CYCLE;
    constexpr int STEPS = (SUBBAND + STEPS_PER_CYCLE - 1) / STEPS_PER_CYCLE;

    wgt_t lowpass[TAPS], highpass[TAPS];
    #pragma HLS ARRAY_PARTITION variable=lowpass type=complete
    #pragma HLS ARRAY_PARTITION variable=highpass type=complete
    load_h: for (int k = 0; k < TAPS; k++) {
        #pragma HLS PIPELINE II=1
        lowpass[k] = LOWPASS[0][0][k];
        highpass[k] = HIGHPASS[0][0][k];
    }

    // Projection weights of band 2m, band 2m + 1, low[m] and high[m].
    wgt_t even_weight[STEPS][STEPS_PER_CYCLE][MID_CHANNELS], odd_weight[STEPS][STEPS_PER_CYCLE][MID_CHANNELS];
    wgt_t low_weight[STEPS][STEPS_PER_CYCLE][MID_CHANNELS], high_weight[STEPS][STEPS_PER_CYCLE][MID_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=even_weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=even_weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=odd_weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=odd_weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=low_weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=low_weight type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=high_weight type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=high_weight type=complete dim=3
    acc_t bias[MID_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=bias type=complete

    load_w: for (int s = 0; s < STEPS; s++) {
        for (int u = 0; u < STEPS_PER_CYCLE; u++) {
            for (int o = 0; o < MID_CHANNELS; o++) {
                #pragma HLS PIPELINE II=1
                const int m = s * STEPS_PER_CYCLE + u;
                const int even = 2 * m < BANDS ? 2 * m : 0;
                const int odd = 2 * m + 1 < BANDS ? 2 * m + 1 : 0;
                const int low = m < SUBBAND ? BANDS + m : 0;
                const int high = m < SUBBAND ? BANDS + SUBBAND + m : 0;
                even_weight[s][u][o] = 2 * m < BANDS ? PROJECTION_WEIGHT[o][even][0][0] : (wgt_t)0;
                odd_weight[s][u][o] = 2 * m + 1 < BANDS ? PROJECTION_WEIGHT[o][odd][0][0] : (wgt_t)0;
                low_weight[s][u][o] = m < SUBBAND ? PROJECTION_WEIGHT[o][low][0][0] : (wgt_t)0;
                high_weight[s][u][o] = m < SUBBAND ? PROJECTION_WEIGHT[o][high][0][0] : (wgt_t)0;
            }
        }
    }
    load_b: for (int o = 0; o < MID_CHANNELS; o++) {
        #pragma HLS PIPELINE II=1
        bias[o] = (acc_t)PROJECTION_BIAS[o] * ((acc_t)1 << BIAS_SHIFT);
    }

    act8_t previous[PAD];   // the last bands of the previous step
    #pragma HLS ARRAY_PARTITION variable=previous type=complete
    acc_t sums[MID_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=sums type=complete

    const int pixels = patch_count<PATCH_HEIGHT, PATCH_WIDTH>(height, width) * PATCH_HEIGHT * PATCH_WIDTH;
    int s = 0;
    pixel_step: for (int t = 0; t < pixels * STEPS; t++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8192 max=19747840
        const vec_t<act8_t, BANDS_PER_CYCLE> bands = in.read();
        act8_t window[PAD + BANDS_PER_CYCLE];
        #pragma HLS ARRAY_PARTITION variable=window type=complete
        for (int k = 0; k < PAD; k++) {
            #pragma HLS UNROLL
            window[k] = (s == 0) ? (act8_t)0 : previous[k];
        }
        for (int k = 0; k < BANDS_PER_CYCLE; k++) {
            #pragma HLS UNROLL
            window[PAD + k] = bands.v[k];
        }
        for (int k = 0; k < PAD; k++) {
            #pragma HLS UNROLL
            previous[k] = window[BANDS_PER_CYCLE + k];
        }

        act_t low[STEPS_PER_CYCLE], high[STEPS_PER_CYCLE];
        #pragma HLS ARRAY_PARTITION variable=low type=complete
        #pragma HLS ARRAY_PARTITION variable=high type=complete
        for (int u = 0; u < STEPS_PER_CYCLE; u++) {
            #pragma HLS UNROLL
            act_t low_sum = 0, high_sum = 0;
            for (int k = 0; k < TAPS; k++) {
                #pragma HLS UNROLL
                low_sum += mul<DWT_DSP, act_t>(window[2 * u + k], lowpass[k]);
                high_sum += mul<DWT_DSP, act_t>(window[2 * u + k], highpass[k]);
            }
            low[u] = low_sum;
            high[u] = high_sum;
        }

        vec_t<act_t, MID_CHANNELS> projected;
        for (int o = 0; o < MID_CHANNELS; o++) {
            #pragma HLS UNROLL
            acc_t delta = 0;
            for (int u = 0; u < STEPS_PER_CYCLE; u++) {
                #pragma HLS UNROLL
                const acc_t from_bands = ((acc_t)mul<PROJECTION_DSP, int16_t>(window[2 * u + PAD], even_weight[s][u][o])
                                        + (acc_t)mul<PROJECTION_DSP, int16_t>(window[2 * u + PAD + 1], odd_weight[s][u][o]))
                                       << BAND_SHIFT;
                const acc_t from_wavelet = mul<PROJECTION_DSP, acc_t>(low[u], low_weight[s][u][o])
                                         + mul<PROJECTION_DSP, acc_t>(high[u], high_weight[s][u][o]);
                delta += from_bands + from_wavelet;
            }
            const acc_t total = (s == 0 ? bias[o] : sums[o]) + delta;
            sums[o] = total;
            projected.v[o] = requant_act<OUT_SHIFT>(total);
        }

        if (s == STEPS - 1) {
            out.write(projected);
            s = 0;
        } else {
            s++;
        }
    }
}
