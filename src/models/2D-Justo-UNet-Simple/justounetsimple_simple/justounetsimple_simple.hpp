#ifndef JUSTOUNETSIMPLE_SIMPLE_HLS_HPP
#define JUSTOUNETSIMPLE_SIMPLE_HLS_HPP

// 2D-Justo-UNet-Simple, the simple version (src/simple/layers.hpp): the golden kernel's
// layers as plain loops with pragmas, a patch at a time in a DATAFLOW region. The same
// answers as justounetsimple_opt, bit for bit. Design: optimizations_implemented/simple.md.

#include <stdint.h>
#include "../../../simple/layers.hpp"
#include "../../../optimized/zscore_int8.hpp"   // generated data: the integer z-score

constexpr int PATCH_SIZE = 32;            // patches are 32 x 32 pixels
constexpr int HALF_PATCH = PATCH_SIZE / 2;
constexpr int CUBE_BANDS = 120;           // bands per pixel in the raw cube
constexpr int MODEL_BANDS = 110;          // bands the model uses (raw 8-117)
constexpr int BASE_CHANNELS = 6;
constexpr int CLASSES = 3;
constexpr int BANDS_PER_WORD = 8;         // one 128-bit word
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int ZSCORE_FRAC_BITS = 5;       // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 11;  // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 16;       // ... of the scores

using simple::cube_word_t;
using simple::score_word_t;

// The argument names become the kernel's register names: keep them.
void justounetsimple_simple(const cube_word_t* din0, score_word_t* dout, int height, int width);

#endif
