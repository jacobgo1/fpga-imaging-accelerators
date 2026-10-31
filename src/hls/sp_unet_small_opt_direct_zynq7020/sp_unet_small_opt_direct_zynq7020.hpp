#ifndef SP_UNET_SMALL_OPT_DIRECT_ZYNQ7020_HLS_HPP
#define SP_UNET_SMALL_OPT_DIRECT_ZYNQ7020_HLS_HPP

#include "sp_unet_opt_zynq7020.hpp"

// SpecPW-UNet small, direct convs, on the Zynq-7020: the chain and its
// sizes are in the .cpp.
// The argument names become the kernel's register names: keep them.
void sp_unet_small_opt_direct_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                                      int height, int width, uint32_t& cycles);

#endif
