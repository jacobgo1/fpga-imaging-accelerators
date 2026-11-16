#ifndef SP_UNET_SIMPLE_HLS_HPP
#define SP_UNET_SIMPLE_HLS_HPP

// SpecPW-UNet (small, medium and large), the simple version (src/simple/layers.hpp): the
// golden kernel's layers as plain loops with pragmas, a patch at a time in a DATAFLOW loop.
// The same answers as sp_unet_SIZE_opt, bit for bit. The chain is in sp_unet_simple_chain.hpp,
// each size's UI x UO table in its .cpp. Design: optimizations_implemented/simple.md.

#include <stdint.h>
#include "../../../../simple/layers.hpp"
#include "../../../../optimized/zscore_int8.hpp"   // generated data: the integer z-score

constexpr int PATCH_SIZE = 32;
constexpr int CUBE_BANDS = 120;
constexpr int MODEL_BANDS = 110;                // raw bands 8-117
constexpr int CLASSES = 3;
constexpr int BANDS_PER_WORD = 8;               // one 128-bit word
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;

constexpr int ZSCORE_FRAC_BITS = 5;             // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 10;        // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 16;             // ... of the scores

using simple::cube_word_t;
using simple::score_word_t;

// A layer's inputs and outputs a cycle, in a size's table.
template<int UI_, int UO_>
struct layer_size {
    static constexpr int UI = UI_, UO = UO_;
};

#endif
