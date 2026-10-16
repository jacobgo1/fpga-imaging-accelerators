#pragma once

// The optimized 1D-Justo-LiuNet's DATAFLOW chain, shared by justoliunet_opt and
// justoliunet_bn_opt (the same network, other weights and activation formats).
// Before including it, a kernel's .cpp includes its model's int8 weights as
// namespace q and its activation formats as namespace fmt
// (justoliunet_opt_formats.hpp). justoliunet_opt.hpp has the design.

#include "justoliunet_opt.hpp"
#include "justoliunet_opt_formats.hpp"
#include "../../optimized/shared/pixel_io.hpp"
#include "../../optimized/shared/conv1d_engine.hpp"
#include "../../optimized/shared/count_cycles.hpp"
#include <type_traits>

namespace jf = justoliunet_opt_formats;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                               decltype(q::conv1_weight)>::type>::type,
                           wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(jf::FIRST_BAND == FIRST * PORT_BANDS && KEPT * PORT_BANDS == 112,
              "the kept bands must be whole words, to the end of each pixel");

// Fraction bits: each layer's sums carry its input's plus its weights'; the
// bias is shifted up to them, the sums rounded down to the next activations'.
constexpr int SUM1 = jf::IN_FRAC + q::conv1_weight_frac;
constexpr int SUM2 = fmt::ACT_FRAC1 + q::conv2_weight_frac;
constexpr int SUM3 = fmt::ACT_FRAC2 + q::conv3_weight_frac;
constexpr int SUM4 = fmt::ACT_FRAC3 + q::conv4_weight_frac;
constexpr int SUMF = fmt::ACT_FRAC4 + q::fc_weight_frac;
// int32 never overflows: conv1's largest |sum| is 2047 x 6 x 128 + bias < 2^21,
// the others' 255 x 108 x 128 + bias < 2^23, the scores' 255 x 48 x 128 x 2^6.

static void justoliunet_opt_dataflow(hls::burst_maxi<jl_raw_t> din0, hls::burst_maxi<jl_raw_t> din1,
                                     hls::burst_maxi<jl_out_t> dout, int height, int width, uint32_t& cycles)
{
#pragma HLS DATAFLOW
    hls::stream<jl_raw_t> s_raw[PORTS];
    hls::stream<map_t<int16_t, BANDS, 1>> s_x("s_x");
    hls::stream<vec_t<uact_t, C1_POS * 6>> s_c1("s_c1");
    hls::stream<map_t<uact_t, 51, 6>> s_p1("s_p1");
    hls::stream<vec_t<uact_t, C2_POS * 12>> s_c2("s_c2");
    hls::stream<map_t<uact_t, 23, 12>> s_p2("s_p2");
    hls::stream<vec_t<uact_t, C3_POS * 18>> s_c3("s_c3");
    hls::stream<map_t<uact_t, 9, 18>> s_p3("s_p3");
    hls::stream<vec_t<uact_t, C4_POS * C4_OCP>> s_c4("s_c4");
    hls::stream<map_t<uact_t, 2, 24>> s_p4("s_p4");
    hls::stream<vec_t<acc_t, CLASSES>> s_scores("s_scores");
    hls::stream<bool> s_finished("s_finished");
// A block of the other port arrives while zscore_pixels takes this one's: let
// each reader run ahead. Everything after it moves one pixel at a time, at the
// same rate: the default depth of 2 keeps them going.
#pragma HLS STREAM variable = s_raw depth = 64
#pragma HLS STREAM variable = s_scores depth = 8

    read_pixels_ahead<WORDS, BLOCK, PORTS, 0>(din0, s_raw[0], height, width);
    read_pixels_ahead<WORDS, BLOCK, PORTS, 1>(din1, s_raw[1], height, width);
    zscore_pixels<WORDS, FIRST, PORT_BANDS, BLOCK, PORTS, jf::PREP_FRAC, jf::IN_MAX, jf::A, jf::B, ZSCORE_DSP>(
        s_raw, s_x, height, width);
    // conv1: 1 -> 6, 102 outputs (110 bands), 12-bit input: not packable.
    conv1d_engine<BANDS, 1, 6, K, 102, C1_POS, 6, SUM1 - q::conv1_bias_frac, SUM1 - fmt::ACT_FRAC1, jf::ACT_MAX,
                  q::conv1_weight, q::conv1_bias, false, C1_DSP>(s_x, s_c1, height, width);
    pool_assemble<102, C1_POS, 6, 6, 51>(s_c1, s_p1, height, width);
    // conv2: 6 -> 12, 46 outputs.
    conv1d_engine<51, 6, 12, K, 46, C2_POS, 12, SUM2 - q::conv2_bias_frac, SUM2 - fmt::ACT_FRAC2, jf::ACT_MAX,
                  q::conv2_weight, q::conv2_bias, C2_PACKED, C2_PACKED>(s_p1, s_c2, height, width);
    pool_assemble<46, C2_POS, 12, 12, 23>(s_c2, s_p2, height, width);
    // conv3: 12 -> 18, 18 outputs.
    conv1d_engine<23, 12, 18, K, 18, C3_POS, 18, SUM3 - q::conv3_bias_frac, SUM3 - fmt::ACT_FRAC3, jf::ACT_MAX,
                  q::conv3_weight, q::conv3_bias, C3_PACKED, C3_PACKED>(s_p2, s_c3, height, width);
    pool_assemble<18, C3_POS, 18, 18, 9>(s_c3, s_p3, height, width);
    // conv4: 18 -> 24, 4 outputs.
    conv1d_engine<9, 18, 24, K, 4, C4_POS, C4_OCP, SUM4 - q::conv4_bias_frac, SUM4 - fmt::ACT_FRAC4, jf::ACT_MAX,
                  q::conv4_weight, q::conv4_bias, C4_PACKED, C4_PACKED>(s_p3, s_c4, height, width);
    pool_assemble<4, C4_POS, C4_OCP, 24, 2>(s_c4, s_p4, height, width);
    // flatten + dense: 48 -> 3, to the scores' OUT_FRAC fraction bits.
    dense_scores<2, 24, CLASSES, DENSE_FEAT, SUMF - q::fc_bias_frac, jf::OUT_FRAC - SUMF,
                 q::fc_weight, q::fc_bias, false>(s_p4, s_scores, height, width);
    store_pixels<CLASSES, 64, 8>(s_scores, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (count_cycles.hpp).
    count_cycles(s_finished, cycles);
}
