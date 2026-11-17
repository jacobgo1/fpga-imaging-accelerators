#ifndef JUSTOLIUNET_SIMPLE_HLS_HPP
#define JUSTOLIUNET_SIMPLE_HLS_HPP

// 1D-Justo-LiuNet, the simple version (src/simple/layers.hpp): the golden kernel's layers,
// four conv1d blocks and the dense layer, each a process on a block of pixels at a time in a
// DATAFLOW loop. A pixel's whole spectrum goes between the layers as one pixel word. The same
// answers as justoliunet_opt (and, with the BatchNorm model's weights, justoliunet_bn_opt) bit
// for bit. The chain is in justoliunet_simple_chain.hpp. Design: optimizations_implemented/simple.md.

#include <stdint.h>
#include "../../../simple/layers.hpp"

constexpr int CUBE_BANDS = 120;
constexpr int BANDS_PER_WORD = 8;
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int MODEL_BANDS = 110;   // raw bands 8-117
constexpr int KERNEL_SIZE = 6;
constexpr int CLASSES = 3;
constexpr int BLOCK = 256;         // pixels a block: a burst of 3 840 words

using simple::cube_word_t;
using simple::score_word_t;

// The argument names become the kernel's register names: keep them.
void justoliunet_simple(const cube_word_t* din0, score_word_t* dout, int height, int width);

#endif
