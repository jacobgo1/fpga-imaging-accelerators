#ifndef SP_UNET_MEDIUM_OPT_DIRECT_ZYNQ7020_HLS_HPP
#define SP_UNET_MEDIUM_OPT_DIRECT_ZYNQ7020_HLS_HPP

#include "../sp_unet_small_opt_direct_zynq7020/sp_unet_opt_zynq7020.hpp"

// Does NOT fit the Zynq-7020 (csynth 2026-10-28: too many LUTs, and on large block
// RAM): bit-exact in C++, kept as a starting point; see optimizations_implemented/
// sp_unet.md, "Medium and large do not fit the 7020". Only small is built for the 7020.
//
// The direct SpecPW-UNet, medium, on the Zynq-7020 (--board zynq7020): the design
// is in sp_unet_small_opt_direct_zynq7020/sp_unet_opt_zynq7020.hpp, its sizes in
// sp_unet_medium_opt_direct_zynq7020.cpp. The same answers as sp_unet_medium_opt_direct.
void sp_unet_medium_opt_direct_zynq7020(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_out_t> dout,
                                        int height, int width, uint32_t& cycles);

#endif // SP_UNET_MEDIUM_OPT_DIRECT_ZYNQ7020_HLS_HPP
