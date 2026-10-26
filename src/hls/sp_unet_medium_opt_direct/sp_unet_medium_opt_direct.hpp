#ifndef SP_UNET_MEDIUM_OPT_DIRECT_HLS_HPP
#define SP_UNET_MEDIUM_OPT_DIRECT_HLS_HPP

#include "../sp_unet_small_opt/sp_unet_opt.hpp"

// The optimized SpecPW-UNet, medium, with direct 3x3 convs, on the ZCU104:
// the design is in sp_unet_small_opt_direct/sp_unet_direct_dataflow.hpp, its
// sizes in sp_unet_medium_opt_direct.cpp. Same ports and answers as sp_unet_medium_opt.
void sp_unet_medium_opt_direct(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_raw_t> din1,
                               hls::burst_maxi<spu_raw_t> din2, hls::burst_maxi<spu_raw_t> din3,
                               hls::burst_maxi<spu_out_t> dout, int height, int width, uint32_t& cycles);

#endif // SP_UNET_MEDIUM_OPT_DIRECT_HLS_HPP
