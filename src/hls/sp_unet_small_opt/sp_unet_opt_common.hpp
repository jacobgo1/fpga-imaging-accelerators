#pragma once

// What the optimized SpecPW-UNet chains share. The kernel's .cpp first includes
// its header, its model's weights as namespace weights and its size table as
// namespace sizes.

#include "../justounetsimple_opt/justounetsimple_prep.hpp"
#include "../../optimized/shared/fixed_point.hpp"

namespace zscore = justounetsimple_prep;   // justounetsimple_opt's preprocessing

// The channel counts, from the weights (int8: a byte each).
constexpr int STEM1 = sizeof(weights::convsp1_bias), STEM2 = sizeof(weights::convsp2_bias);
constexpr int STEM3 = sizeof(weights::convsp3_bias);
constexpr int DOUBLE1 = sizeof(weights::double1_block_0_bias), DOUBLE2 = sizeof(weights::double2_block_0_bias);
constexpr int DOUBLE3 = sizeof(weights::double3_block_0_bias), DOUBLE4 = sizeof(weights::double4_block_0_bias);

constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = zscore::FIRST_BAND / BANDS_PER_WORD;
constexpr int CONVSP1_CHUNKS = (MODEL_BANDS + sizes::CONVSP1_BANDS_PER_CYCLE - 1) / sizes::CONVSP1_BANDS_PER_CYCLE;

// A layer's sums have its input's fraction bits plus its weights'; the bias is
// shifted up to them, the sums rounded down to the output's.
template<int IN_FRAC_BITS, int WEIGHT_FRAC_BITS, int BIAS_FRAC_BITS, int OUT_FRAC_BITS>
struct layer_fracs {
    static constexpr int SUM = IN_FRAC_BITS + WEIGHT_FRAC_BITS;
    static constexpr int BIAS_SHIFT = SUM - BIAS_FRAC_BITS, SHIFT = SUM - OUT_FRAC_BITS;
};
#define LAYER_FRACS(name, layer, in_frac, out_frac) \
    typedef layer_fracs<in_frac, weights::layer##_weight_frac, weights::layer##_bias_frac, out_frac> name
LAYER_FRACS(CONVSP1_FRACS, convsp1, ZSCORE_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(CONVSP2_FRACS, convsp2, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(CONVSP3_FRACS, convsp3, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE1_CONV1_FRACS, double1_block_0, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE1_CONV2_FRACS, double1_block_3, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE2_CONV1_FRACS, double2_block_0, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE2_CONV2_FRACS, double2_block_3, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(UP1_FRACS, up1_1, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE3_CONV1_FRACS, double3_block_0, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE3_CONV2_FRACS, double3_block_3, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(UP2_FRACS, up2_1, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE4_CONV1_FRACS, double4_block_0, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(DOUBLE4_CONV2_FRACS, double4_block_3, ACTIVATION_FRAC_BITS, ACTIVATION_FRAC_BITS);
LAYER_FRACS(FINAL_FRACS, final, ACTIVATION_FRAC_BITS, SCORE_FRAC_BITS);
#undef LAYER_FRACS
