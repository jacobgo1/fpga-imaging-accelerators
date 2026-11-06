#pragma once

// What the uint8 SpecPW-UNet chains (sp_unet_SIZE_opt_u8) share. The kernel's .cpp first
// includes its header, its model's weights as namespace weights, its size table as
// namespace sizes and its activations' fraction bits (specpw_unet_act8_fracs.hpp) as
// namespace act8.

#include "../sp_unet_small_opt/sp_unet_opt_common.hpp"   // channel counts, layer_fracs

// Each layer's fractions: its input's fraction bits (the layer before it, or the z-scores'),
// its own. A layer's activation a means a * 2^-F, 0 .. 255; F is fixed, the same for every
// image (tools/specpw_unet_act8.py --write chose it). The scores keep SCORE_FRAC_BITS, more
// than final's sums have, so its SHIFT is negative: store_scores shifts them left, exactly.
namespace u8 {
#define U8_LAYER_FRACS(name, layer, in_frac, out_frac) \
    typedef layer_fracs<in_frac, weights::layer##_weight_frac, weights::layer##_bias_frac, out_frac> name
U8_LAYER_FRACS(CONVSP1, convsp1, ZSCORE_FRAC_BITS, act8::convsp1);
U8_LAYER_FRACS(CONVSP2, convsp2, act8::convsp1, act8::convsp2);
U8_LAYER_FRACS(CONVSP3, convsp3, act8::convsp2, act8::convsp3);
U8_LAYER_FRACS(DOUBLE1_CONV1, double1_block_0, act8::convsp3, act8::double1_block_0);
U8_LAYER_FRACS(DOUBLE1_CONV2, double1_block_3, act8::double1_block_0, act8::double1_block_3);
U8_LAYER_FRACS(DOUBLE2_CONV1, double2_block_0, act8::double1_block_3, act8::double2_block_0);
U8_LAYER_FRACS(DOUBLE2_CONV2, double2_block_3, act8::double2_block_0, act8::double2_block_3);
U8_LAYER_FRACS(UP1, up1_1, act8::double2_block_3, act8::up1_1);
U8_LAYER_FRACS(DOUBLE3_CONV1, double3_block_0, act8::up1_1, act8::double3_block_0);
U8_LAYER_FRACS(DOUBLE3_CONV2, double3_block_3, act8::double3_block_0, act8::double3_block_3);
U8_LAYER_FRACS(UP2, up2_1, act8::double3_block_3, act8::up2_1);
U8_LAYER_FRACS(DOUBLE4_CONV1, double4_block_0, act8::up2_1, act8::double4_block_0);
U8_LAYER_FRACS(DOUBLE4_CONV2, double4_block_3, act8::double4_block_0, act8::double4_block_3);
U8_LAYER_FRACS(FINAL, final, act8::double4_block_3, SCORE_FRAC_BITS);
#undef U8_LAYER_FRACS
}

static_assert(act8::up2_1 == act8::double1_block_3, "the concat joins up2 and the skip: one scale");
