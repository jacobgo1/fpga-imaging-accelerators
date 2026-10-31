#ifndef SP_UNET_SMALL_OPT_DIRECT_HLS_HPP
#define SP_UNET_SMALL_OPT_DIRECT_HLS_HPP

#include "../sp_unet_small_opt/sp_unet_opt.hpp"

// SpecPW-UNet small, direct convs, on the ZCU104: the chain and its sizes are in the .cpp. Without
// Winograd: every 3x3 conv done directly (conv3x3_grouped), the maps in raster order, the
// pooling and upsampling in justounetsimple_opt's ReLU modules. The same answers; simpler,
// slower for the same DSPs.
// The argument names become the kernel's register names: keep them.
void sp_unet_small_opt_direct(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                             hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                             hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
