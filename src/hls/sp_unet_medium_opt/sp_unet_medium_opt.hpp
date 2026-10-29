#ifndef SP_UNET_MEDIUM_OPT_HLS_HPP
#define SP_UNET_MEDIUM_OPT_HLS_HPP

#include "../sp_unet_small_opt/sp_unet_opt.hpp"

// SpecPW-UNet medium, Winograd, on the ZCU104: the chain is sp_unet_opt_dataflow.hpp, the sizes in the .cpp.
// The argument names become the kernel's register names: keep them.
void sp_unet_medium_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                       hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                       hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
