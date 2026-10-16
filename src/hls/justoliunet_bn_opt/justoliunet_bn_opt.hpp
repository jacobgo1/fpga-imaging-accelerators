#ifndef JUSTOLIUNET_BN_OPT_HLS_HPP
#define JUSTOLIUNET_BN_OPT_HLS_HPP

// justoliunet_opt (read its header) with the BatchNorm model's weights: the
// trained BatchNorm after conv1 is folded into conv1 (tools/quantize_weights.py),
// so the network, the formats of the data in and out and the hardware are the
// same; the weights and the activations' fraction bits are this model's.
#include "../justoliunet_opt/justoliunet_opt.hpp"

void justoliunet_bn_opt(hls::burst_maxi<jl_raw_t> din0, hls::burst_maxi<jl_raw_t> din1,
                        hls::burst_maxi<jl_out_t> dout, int height, int width, uint32_t& cycles);

#endif // JUSTOLIUNET_BN_OPT_HLS_HPP
