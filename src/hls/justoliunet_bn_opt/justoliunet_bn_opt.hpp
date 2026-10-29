#ifndef JUSTOLIUNET_BN_OPT_HLS_HPP
#define JUSTOLIUNET_BN_OPT_HLS_HPP

// justoliunet_opt with the BatchNorm model's weights, the BatchNorm folded into
// conv1: the same hardware, other weights and activation fraction bits.
#include "../justoliunet_opt/justoliunet_opt.hpp"

void justoliunet_bn_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                        hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
