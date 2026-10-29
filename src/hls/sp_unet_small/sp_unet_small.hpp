#ifndef SP_UNET_SMALL_HLS_HPP
#define SP_UNET_SMALL_HLS_HPP

#include "../../golden/specpw_unet_quant_golden.hpp"

// Golden SpecPW-UNet, small (stem 24-12-12, blocks 12 / 24 / 12 / 8): n z-scored 32 x 32 patches a start, 3 scores
// per pixel out. Plain float loops, with the int8 weights compiled in.

using data_t = float;
using model_size = specpw_unet_small;

constexpr int GOLDEN_PATCH_SIZE = 32;
constexpr int GOLDEN_BANDS = 110;
constexpr int GOLDEN_CLASSES = 3;

// din: [n][32][32][110] float. dout: [n][32][32][3] float (0 cloud, 1 land, 2 sea).
void sp_unet_small(const data_t *din, data_t *dout, int n);

#endif
