#ifndef JUSTOLIUNET_BN_SIMPLE_HLS_HPP
#define JUSTOLIUNET_BN_SIMPLE_HLS_HPP

// justoliunet_simple with the BatchNorm model's weights, the BatchNorm folded into conv1:
// the same hardware, other weights and activation fraction bits.
#include "../../1D-Justo-LiuNet/justoliunet_simple/justoliunet_simple.hpp"

// The argument names become the kernel's register names: keep them.
void justoliunet_bn_simple(const cube_word_t* din0, score_word_t* dout, int height, int width);

#endif
