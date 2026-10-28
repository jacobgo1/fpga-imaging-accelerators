#pragma once

// What the optimized SpecPW-UNet kernels' DATAFLOW chains share (the Winograd one,
// sp_unet_opt_dataflow.hpp, and the direct one, sp_unet_direct_dataflow.hpp): the
// checks on the weights and the preprocessing, the channel counts, the fraction
// bits. Before including it, a kernel's .cpp includes its header and its model's int8 weights as
// namespace q and its table of sizes as namespace cfg.

// The kernel's header comes first (sp_unet_opt.hpp, or sp_unet_opt_zynq7020.hpp):
// it sets the ports, the word width and the formats this file checks against.
#include "../justounetsimple_opt/justounetsimple_prep.hpp"
#include "../../optimized/shared/fixed_point.hpp"
#include <type_traits>

// The preprocessing is justounetsimple_opt's (tools/specpw_unet_model.py uses
// its preprocess_fixed): the same integer z-score constants.
namespace prep = justounetsimple_prep;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                               decltype(q::convsp1_weight)>::type>::type,
                           wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(prep::IN_FRAC == IN_FRAC && prep::IN_MAX == 127,
              "justounetsimple_prep.hpp is out of date: run python tools/justounetsimple_model.py");

// The channel counts, from the weights (int8: one byte each).
constexpr int S1 = sizeof(q::convsp1_bias), S2 = sizeof(q::convsp2_bias), S3 = sizeof(q::convsp3_bias);
constexpr int C1 = sizeof(q::double1_block_0_bias), C2 = sizeof(q::double2_block_0_bias);
constexpr int C3 = sizeof(q::double3_block_0_bias), C4 = sizeof(q::double4_block_0_bias);
static_assert(sizeof(q::convsp1_weight) == S1 * IN_CH && sizeof(q::final_bias) == CLASSES &&
                  sizeof(q::double4_block_0_weight) == C4 * (C3 + C1) * 9,
              "the weights are not SpecPW-UNet's");

constexpr int WORDS = RAW_BANDS / PORT_BANDS;        // 15 words per pixel in the cube
constexpr int FIRST = prep::FIRST_BAND / PORT_BANDS; // word 0 (raw bands 0-7) is dropped
constexpr int KEPT = WORDS - FIRST;                  // words 1-14: kept bands 0-111
constexpr int CHUNKS = (IN_CH + cfg::P1 - 1) / cfg::P1;   // convsp1's chunks a pixel
static_assert(RAW_BANDS % PORT_BANDS == 0 && cfg::P1 % PORT_BANDS == 0, "a chunk is whole words");
static_assert(prep::FIRST_BAND % PORT_BANDS == 0 && KEPT * PORT_BANDS >= IN_CH &&
                  (KEPT * PORT_BANDS + cfg::P1 - 1) / cfg::P1 == CHUNKS &&
                  sizeof(prep::A) / sizeof(prep::A[0]) == KEPT * PORT_BANDS,
              "the kept bands must be whole words, to the end of each pixel, in convsp1's chunks");

// A layer's fraction bits: its sums carry its input's plus its weights'; the
// bias is shifted up to them, the sums rounded off to the next activations'.
template<int IN_F, int W_F, int B_F, int OUT_F>
struct spu_fracs {
    static constexpr int SUM = IN_F + W_F, BIAS_SHIFT = SUM - B_F, SHIFT = SUM - OUT_F;
    static_assert(BIAS_SHIFT >= 0, "a bias has more fraction bits than its layer's sums");
    static_assert(SHIFT >= 0, "a requantization would have to add bits");
};
#define SPU_FRACS(name, layer, in_frac, out_frac) \
    typedef spu_fracs<in_frac, q::layer##_weight_frac, q::layer##_bias_frac, out_frac> name
SPU_FRACS(F_SP1, convsp1, IN_FRAC, ACT_FRAC);
SPU_FRACS(F_SP2, convsp2, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_SP3, convsp3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D1A, double1_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D1B, double1_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D2A, double2_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D2B, double2_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_UP1, up1_1, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D3A, double3_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D3B, double3_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_UP2, up2_1, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D4A, double4_block_0, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_D4B, double4_block_3, ACT_FRAC, ACT_FRAC);
SPU_FRACS(F_FIN, final, ACT_FRAC, OUT_FRAC);
#undef SPU_FRACS
