#pragma once

// The Zynq-7020 1D-Justo-LiuNet's DATAFLOW chain, shared by
// justoliunet_opt_zynq7020 and justoliunet_bn_opt_zynq7020 (the same network,
// other weights and activation formats). Before including it, a kernel's .cpp
// includes its model's int8 weights as namespace q and its activation formats as
// namespace fmt (../justoliunet_opt/justoliunet_opt_formats.hpp, the ZCU104
// kernels' formats). justoliunet_opt_zynq7020.hpp has the design.

#include "justoliunet_opt_zynq7020.hpp"
#include "../justoliunet_opt/justoliunet_opt_formats.hpp"
#include "../../optimized/shared/pixel_io.hpp"
#include "../../optimized/shared/conv1d_engine.hpp"
#include "../../optimized/shared/conv1d_stream.hpp"
#include "../../optimized/shared/count_cycles.hpp"
#include <type_traits>

namespace jf = justoliunet_opt_formats;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                               decltype(q::conv1_weight)>::type>::type,
                           wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(jf::FIRST_BAND == FIRST * PORT_BANDS && KEPT * PORT_BANDS == 112,
              "the kept bands must be whole words, to the end of each pixel");

// Fraction bits, as in justoliunet_opt_dataflow.hpp.
constexpr int SUM1 = jf::IN_FRAC + q::conv1_weight_frac;
constexpr int SUM2 = fmt::ACT_FRAC1 + q::conv2_weight_frac;
constexpr int SUM3 = fmt::ACT_FRAC2 + q::conv3_weight_frac;
constexpr int SUM4 = fmt::ACT_FRAC3 + q::conv4_weight_frac;
constexpr int SUMF = fmt::ACT_FRAC4 + q::fc_weight_frac;

static void justoliunet_opt_zynq7020_dataflow(hls::burst_maxi<jl_raw_t> din0, hls::burst_maxi<jl_out_t> dout,
                                              int height, int width, uint32_t& cycles)
{
#pragma HLS DATAFLOW
    hls::stream<jl_raw_t> s_raw[PORTS];
    hls::stream<map_t<int16_t, BANDS, 1>> s_x("s_x");
    hls::stream<vec_t<uact_t, 6>> s_c1("s_c1"), s_p1("s_p1");
    hls::stream<vec_t<uact_t, 12>> s_p2("s_p2");
    hls::stream<vec_t<uact_t, 18>> s_p3("s_p3");
    hls::stream<vec_t<uact_t, 24>> s_p4("s_p4");
    hls::stream<vec_t<acc_t, CLASSES>> s_scores("s_scores");
    hls::stream<bool> s_finished("s_finished");
// The reader's bursts arrive in blocks; let it run ahead. A layer reads its
// first 5 positions of a pixel back to back (its window): room for them, so the
// one before it does not wait.
#pragma HLS STREAM variable = s_raw depth = 64
#pragma HLS STREAM variable = s_p1 depth = 8
#pragma HLS STREAM variable = s_p2 depth = 8
#pragma HLS STREAM variable = s_p3 depth = 8
#pragma HLS STREAM variable = s_scores depth = 8

    read_pixels_ahead<WORDS, BLOCK, PORTS, 0>(din0, s_raw[0], height, width);
    zscore_pixels<WORDS, FIRST, PORT_BANDS, BLOCK, PORTS, jf::PREP_FRAC, jf::IN_MAX, jf::A, jf::B, ZSCORE_DSP>(
        s_raw, s_x, height, width);
    // conv1: 1 -> 6, 102 outputs, one a cycle, 12-bit input: 36 LUT multipliers.
    conv1d_engine<BANDS, 1, 6, K, 102, 1, 6, SUM1 - q::conv1_bias_frac, SUM1 - fmt::ACT_FRAC1, jf::ACT_MAX,
                  q::conv1_weight, q::conv1_bias, false, false>(s_x, s_c1, height, width);
    pool_pairs<102, 6>(s_c1, s_p1, height, width);
    // conv2: 6 -> 12, 46 outputs; 51 positions in, 23 out.
    conv1d_stream<51, 6, 12, K, 46, C2_OCP, C2_ICP, SUM2 - q::conv2_bias_frac, SUM2 - fmt::ACT_FRAC2, jf::ACT_MAX,
                  q::conv2_weight, q::conv2_bias, C2_PACKED, C2_PACKED>(s_p1, s_p2, height, width);
    // conv3: 12 -> 18, 18 outputs; 23 in, 9 out.
    conv1d_stream<23, 12, 18, K, 18, C3_OCP, C3_ICP, SUM3 - q::conv3_bias_frac, SUM3 - fmt::ACT_FRAC3, jf::ACT_MAX,
                  q::conv3_weight, q::conv3_bias, C3_PACKED, C3_PACKED>(s_p2, s_p3, height, width);
    // conv4: 18 -> 24, 4 outputs; 9 in, 2 out.
    conv1d_stream<9, 18, 24, K, 4, C4_OCP, C4_ICP, SUM4 - q::conv4_bias_frac, SUM4 - fmt::ACT_FRAC4, jf::ACT_MAX,
                  q::conv4_weight, q::conv4_bias, C4_PACKED, C4_PACKED>(s_p3, s_p4, height, width);
    // flatten + dense: 2 positions x 24 -> 3, to the scores' OUT_FRAC fraction bits.
    dense_stream<2, 24, CLASSES, DENSE_FEAT, SUMF - q::fc_bias_frac, jf::OUT_FRAC - SUMF,
                 q::fc_weight, q::fc_bias, false>(s_p4, s_scores, height, width);
    store_pixels<CLASSES, 64, 8>(s_scores, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (count_cycles.hpp).
    count_cycles(s_finished, cycles);
}
