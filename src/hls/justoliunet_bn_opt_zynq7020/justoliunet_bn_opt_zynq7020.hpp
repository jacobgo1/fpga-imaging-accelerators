#ifndef JUSTOLIUNET_BN_OPT_ZYNQ7020_HLS_HPP
#define JUSTOLIUNET_BN_OPT_ZYNQ7020_HLS_HPP

// justoliunet_opt_zynq7020 with the BatchNorm model's weights: the same hardware.
#include "../justoliunet_opt_zynq7020/justoliunet_opt_zynq7020.hpp"

void justoliunet_bn_opt_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                                 int height, int width, uint32_t& cycles);

#endif
