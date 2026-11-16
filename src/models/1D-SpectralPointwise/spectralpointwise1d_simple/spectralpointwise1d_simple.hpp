#ifndef SPECTRALPOINTWISE1D_SIMPLE_HLS_HPP
#define SPECTRALPOINTWISE1D_SIMPLE_HLS_HPP

// 1D-SpectralPointwise, the simple version (src/simple/layers.hpp): three 1x1 convs on a
// block of pixels at a time, in a DATAFLOW loop. The same answers as spectralpointwise1d_opt,
// bit for bit. Design: optimizations_implemented/simple.md.

#include <stdint.h>
#include "../../../simple/layers.hpp"

constexpr int CUBE_BANDS = 120;
constexpr int BANDS_PER_WORD = 8;
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int MODEL_BANDS = 110;   // raw bands 8-117
constexpr int CLASSES = 3;
constexpr int BLOCK = 256;         // pixels a block: a burst of 3 840 words

using simple::cube_word_t;
using simple::score_word_t;

// The argument names become the kernel's register names: keep them.
void spectralpointwise1d_simple(const cube_word_t* din0, score_word_t* dout, int height, int width);

#endif
