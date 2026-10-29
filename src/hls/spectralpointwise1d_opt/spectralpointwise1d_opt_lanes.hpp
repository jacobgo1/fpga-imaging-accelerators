#pragma once

// One lane: a copy of the network, its three layers as processes on lane LANE's
// pixels. Shared by every board's kernel; the kernel's header comes first.

#include "../../../weights/quantized/spectralpointwise1d/spectralpointwise1d_int8.hpp"
#include "../justoliunet_opt/justoliunet_opt_formats.hpp"
#include "spectralpointwise1d_opt_formats.hpp"
#include "../../optimized/shared/pixel_io.hpp"
#include "../../optimized/shared/zscore_pointwise.hpp"
#include "../../optimized/shared/pointwise1d.hpp"
#include "../../optimized/shared/count_cycles.hpp"

namespace weights = spectralpointwise1d_int8;
namespace shared_formats = justoliunet_opt_formats;    // the input and the scores, as 1D-Justo-LiuNet's
namespace model_formats = spectralpointwise1d_formats; // the activations

// A layer's sums have its input's fraction bits plus its weights'.
constexpr int CONVSP1_SUM_FRAC_BITS = shared_formats::IN_FRAC + weights::convsp1_weight_frac;
constexpr int CONVSP2_SUM_FRAC_BITS = model_formats::ACT_FRAC1 + weights::convsp2_weight_frac;
constexpr int CONVSP3_SUM_FRAC_BITS = model_formats::ACT_FRAC2 + weights::convsp3_weight_frac;

template<int LANE> using lane_pixels = pixel_lane<BLOCK_PIXELS, LANES, LANE>;

// INLINE: the wrapper goes, the module stays its own process.
template<int LANE>
void lane_convsp1(hls::stream<cube_word_t>& in, hls::stream<vec_t<uact_t, 24> >& out, int height, int width) {
    #pragma HLS INLINE
    zscore_pointwise<WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD, shared_formats::PREP_FRAC,
                     shared_formats::IN_MAX, shared_formats::A, shared_formats::B, 110, 24,
                     CONVSP1_SUM_FRAC_BITS - weights::convsp1_bias_frac, CONVSP1_SUM_FRAC_BITS - model_formats::ACT_FRAC1,
                     shared_formats::ACT_MAX, weights::convsp1_weight, weights::convsp1_bias,
                     ZSCORE_ON_DSPS, CONVSP1_ON_DSPS, lane_pixels<LANE> >(in, out, height, width);
}

template<int LANE>
void lane_convsp2(hls::stream<vec_t<uact_t, 24> >& in, hls::stream<vec_t<uact_t, 12> >& out, int height, int width) {
    #pragma HLS INLINE
    pointwise1d<24, 12, CONVSP2_OUT_PER_CYCLE, true, CONVSP2_SUM_FRAC_BITS - weights::convsp2_bias_frac,
                CONVSP2_SUM_FRAC_BITS - model_formats::ACT_FRAC2, shared_formats::ACT_MAX,
                weights::convsp2_weight, weights::convsp2_bias, CONVSP2_ON_DSPS, lane_pixels<LANE> >(
        in, out, height, width);
}

// The head: the scores.
template<int LANE>
void lane_convsp3(hls::stream<vec_t<uact_t, 12> >& in, hls::stream<vec_t<acc_t, CLASSES> >& out,
                  int height, int width) {
    #pragma HLS INLINE
    pointwise1d<12, CLASSES, CONVSP3_OUT_PER_CYCLE, false, CONVSP3_SUM_FRAC_BITS - weights::convsp3_bias_frac,
                shared_formats::OUT_FRAC - CONVSP3_SUM_FRAC_BITS, shared_formats::ACT_MAX,
                weights::convsp3_weight, weights::convsp3_bias, CONVSP3_ON_DSPS, lane_pixels<LANE> >(
        in, out, height, width);
}
