#pragma once

// The optimized 1D-SpectralPointwise's lane: one copy of the network, its three
// layers as DATAFLOW processes, shared by every board's kernel (ZCU104, Zynq-7030,
// Zynq-7020; they differ in the words, 8 or 4 bands, the lanes and where the
// multipliers go). Before including it, a kernel's .cpp includes its own header,
// which sets those. spectralpointwise1d_opt.hpp has the design.

#include "../../../weights/quantized/spectralpointwise1d/spectralpointwise1d_int8.hpp"
#include "../justoliunet_opt/justoliunet_opt_formats.hpp"
#include "spectralpointwise1d_opt_formats.hpp"
#include "../../optimized/shared/pixel_io.hpp"
#include "../../optimized/shared/zscore_pointwise.hpp"
#include "../../optimized/shared/pointwise1d.hpp"
#include "../../optimized/shared/count_cycles.hpp"
#include <type_traits>

namespace q = spectralpointwise1d_int8;
namespace jf = justoliunet_opt_formats;           // the input: 12-bit z-score, as 1D-Justo-LiuNet's
namespace fmt = spectralpointwise1d_formats;      // the activations
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                               decltype(q::convsp1_weight)>::type>::type,
                           wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(jf::FIRST_BAND == FIRST * PORT_BANDS && KEPT * PORT_BANDS == 112,
              "the kept bands must be whole words, to the end of each pixel");

// Fraction bits: each layer's sums carry its input's plus its weights'; the
// bias is shifted up to them, the sums rounded down to the next activations'.
constexpr int SUM1 = jf::IN_FRAC + q::convsp1_weight_frac;
constexpr int SUM2 = fmt::ACT_FRAC1 + q::convsp2_weight_frac;
constexpr int SUM3 = fmt::ACT_FRAC2 + q::convsp3_weight_frac;
// int32 never overflows: convsp1's largest |sum| is 2047 x 110 x 128 + bias < 2^25,
// the others' 255 x 24 x 128 + bias < 2^20, the scores' 255 x 12 x 128 x 2^5.

// A lane's three layers, each one DATAFLOW process (INLINE: the wrapper goes,
// the module's call stays), on lane L's pixels.
template<int L> using lane_pixels = pixel_lane<BLOCK, LANES, L>;

template<int L>
void spw_convsp1(hls::stream<spw_raw_t>& in, hls::stream<vec_t<uact_t, 24> >& out, int height, int width) {
    #pragma HLS INLINE
    zscore_pointwise<WORDS, FIRST, PORT_BANDS, jf::PREP_FRAC, jf::IN_MAX, jf::A, jf::B, 110, 24,
                     SUM1 - q::convsp1_bias_frac, SUM1 - fmt::ACT_FRAC1, jf::ACT_MAX,
                     q::convsp1_weight, q::convsp1_bias, ZSCORE_DSP, SP1_DSP, lane_pixels<L> >(in, out, height, width);
}

template<int L>
void spw_convsp2(hls::stream<vec_t<uact_t, 24> >& in, hls::stream<vec_t<uact_t, 12> >& out, int height, int width) {
    #pragma HLS INLINE
    pointwise1d<24, 12, SP2_PO, true, SUM2 - q::convsp2_bias_frac, SUM2 - fmt::ACT_FRAC2, jf::ACT_MAX,
                q::convsp2_weight, q::convsp2_bias, SP2_DSP, lane_pixels<L> >(in, out, height, width);
}

// The head: the scores at the output's OUT_FRAC fraction bits.
template<int L>
void spw_convsp3(hls::stream<vec_t<uact_t, 12> >& in, hls::stream<vec_t<acc_t, CLASSES> >& out, int height, int width) {
    #pragma HLS INLINE
    pointwise1d<12, CLASSES, SP3_PO, false, SUM3 - q::convsp3_bias_frac, jf::OUT_FRAC - SUM3, jf::ACT_MAX,
                q::convsp3_weight, q::convsp3_bias, SP3_DSP, lane_pixels<L> >(in, out, height, width);
}
