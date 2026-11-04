#ifndef JUSTOUNETSIMPLE_OPT_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_HLS_HPP
#include <stdint.h>
#include "../../../common/hls_vector_shim.hpp"
#include "../../../common/hls_burst_maxi_shim.hpp"
#include "../../../optimized/shared/zscore_int8.hpp"

namespace zscore = zscore_int8;

constexpr int PATCH_SIZE = 32;            // patches are 32 x 32 pixels
constexpr int CUBE_BANDS = 120;           // bands per pixel in the raw cube
constexpr int MODEL_BANDS = 110;          // bands the model uses (raw 8-117)
constexpr int BASE_CHANNELS = 6;
constexpr int CLASSES = 3;
constexpr int READ_PORTS = 4;
constexpr int BANDS_PER_WORD = 8;         // one 128-bit word
constexpr int CONV1_BANDS_PER_CYCLE = 32;
constexpr int ZSCORE_FRAC_BITS = 5;       // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 11;  // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 16;       // ... of the scores

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

constexpr int HALF_PATCH = PATCH_SIZE / 2;
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = zscore::FIRST_BAND / BANDS_PER_WORD;
constexpr bool ZSCORE_ON_DSPS = true;
constexpr bool CONV2_TO_4_ON_DSPS = true;

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 18, DSP_PACK_GROUP = 8;   // DSP48E2, 27-bit input: two int8 weights 18 bits apart, groups of 8
constexpr bool HAS_URAM = true;                          // the patch buffer in UltraRAM

// The argument names become the kernel's register names: keep them.
void justounetsimple_opt(const cube_word_t* din0, const cube_word_t* din1,
                         const cube_word_t* din2, const cube_word_t* din3,
                         hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
