#ifndef SP_UNET_LARGE_OPT_HLS_HPP
#define SP_UNET_LARGE_OPT_HLS_HPP

#include "../sp_unet_small_opt/sp_unet_opt.hpp"

// The optimized SpecPW-UNet, large, on the ZCU104: the design is in
// sp_unet_small_opt/sp_unet_opt.hpp, its sizes in sp_unet_large_opt.cpp.
void sp_unet_large_opt(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_raw_t> din1,
                       hls::burst_maxi<spu_raw_t> din2, hls::burst_maxi<spu_raw_t> din3,
                       hls::burst_maxi<spu_out_t> dout, int height, int width, uint32_t& cycles);

#endif // SP_UNET_LARGE_OPT_HLS_HPP
