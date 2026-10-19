#ifndef JUSTOLIUNET_BN_OPT_ZYNQ7020_HLS_HPP
#define JUSTOLIUNET_BN_OPT_ZYNQ7020_HLS_HPP

// justoliunet_opt_zynq7020 (read its header) with the BatchNorm model's weights:
// the trained BatchNorm after conv1 is folded into conv1, so the network, the
// formats and the hardware are the same; the weights and the activations'
// fraction bits are this model's (as justoliunet_bn_opt on the ZCU104).
#include "../justoliunet_opt_zynq7020/justoliunet_opt_zynq7020.hpp"

void justoliunet_bn_opt_zynq7020(hls::burst_maxi<jl_raw_t> din0, hls::burst_maxi<jl_out_t> dout,
                                 int height, int width, uint32_t& cycles);

#endif // JUSTOLIUNET_BN_OPT_ZYNQ7020_HLS_HPP
