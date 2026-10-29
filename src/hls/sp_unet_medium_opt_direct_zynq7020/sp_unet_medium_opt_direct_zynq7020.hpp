#ifndef SP_UNET_MEDIUM_OPT_DIRECT_ZYNQ7020_HLS_HPP
#define SP_UNET_MEDIUM_OPT_DIRECT_ZYNQ7020_HLS_HPP

#include "../sp_unet_small_opt_direct_zynq7020/sp_unet_opt_zynq7020.hpp"

// SpecPW-UNet medium, direct convs, on the Zynq-7020: the chain is sp_unet_direct_zynq7020_dataflow.hpp,
// the sizes in the .cpp.
// It does not fit the 7020 (too many LUTs); kept as a starting point, see optimizations_implemented/sp_unet.md.
// The argument names become the kernel's register names: keep them.
void sp_unet_medium_opt_direct_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                                       int height, int width, uint32_t& cycles);

#endif
