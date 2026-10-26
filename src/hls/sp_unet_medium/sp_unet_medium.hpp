#ifndef SP_UNET_MEDIUM_HLS_HPP
#define SP_UNET_MEDIUM_HLS_HPP

#include "../../golden/specpw_unet_quant_golden.hpp"

// The quantized SpecPW-UNet, medium (stem 24-16-12, 24 / 48 / 24 / 12 channels), on n 32 x 32 patches per start
// (src/golden/specpw_unet_quant_golden.hpp): a golden baseline, plain loops in
// float. The int8 weights and their shifts come from
// weights/quantized/sp_unet_medium/ and are compiled into the kernel as constant
// arrays (on-chip ROM), so the interface is only the patches in and the scores
// out. Changing the weights means rebuilding the bitstream.
//
// The host cuts the capture into 32 x 32 patches (the last row and column
// repeated to fill the edge ones, tools/patching.py) and preprocesses them as in
// training: the 110 kept bands (raw 8-117), z-scored with mu_sd.txt.
//
// din:  [n][32][32][110] float, patch by patch, HWC.
// dout: [n][32][32][3] float, the scores (0 cloud, 1 land, 2 sea; highest = class).

using data_t = float;
using spu_size = specpw_unet_medium;

#define SPU_H      32
#define SPU_W      32
#define SPU_IN_CH  110
#define SPU_OUT_CH 3

void sp_unet_medium(const data_t *din, data_t *dout, int n);

#endif // SP_UNET_MEDIUM_HLS_HPP
