#ifndef WAVELETCNN_SIMPLE_HLS_HPP
#define WAVELETCNN_SIMPLE_HLS_HPP

// WaveletCNN, the simple version (src/simple/layers.hpp): the golden kernel's layers (the
// wavelet transform along the bands, the 1x1 projection, the 3x3 conv, the 1x1 head) as
// plain loops with pragmas, a patch at a time in a DATAFLOW loop. The same answers as
// waveletcnn_opt, bit for bit. Design: optimizations_implemented/simple.md.

#include <stdint.h>
#include "../../../simple/layers.hpp"
#include "../../../optimized/zscore_int8.hpp"   // generated data: the integer z-score

constexpr int PATCH_SIZE = 32;
constexpr int CUBE_BANDS = 120;
constexpr int MODEL_BANDS = 110;                // raw bands 8-117
constexpr int DWT_TAPS = 8;                     // db4
constexpr int SUBBAND_LENGTH = (MODEL_BANDS + DWT_TAPS - 1) / 2;
constexpr int WAVELET_CHANNELS = MODEL_BANDS + 2 * SUBBAND_LENGTH;
constexpr int MID_CHANNELS = 12;
constexpr int CLASSES = 3;
constexpr int BANDS_PER_WORD = 8;               // one 128-bit word
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;

constexpr int ZSCORE_FRAC_BITS = 5;             // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 9;         // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 15;             // ... of the scores

using simple::cube_word_t;
using simple::score_word_t;

// The argument names become the kernel's register names: keep them.
void waveletcnn_simple(const cube_word_t* din0, score_word_t* dout, int height, int width);

#endif
