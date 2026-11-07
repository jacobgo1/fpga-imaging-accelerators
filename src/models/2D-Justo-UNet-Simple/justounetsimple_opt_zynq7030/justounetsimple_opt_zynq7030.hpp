#ifndef JUSTOUNETSIMPLE_OPT_ZYNQ7030_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_ZYNQ7030_HLS_HPP

#include <stdint.h>
#include "../../../common/hls_vector_shim.hpp"
#include "../../../common/hls_burst_maxi_shim.hpp"
#include "../../../optimized/zscore_int8.hpp"

namespace zscore = zscore_int8;

// justounetsimple_opt for the Zynq-7030 (--board zynq7030): the 7020 version made wider,
// conv1 on 324 of the 400 DSPs, conv2-4 on 18 each, four 64-bit read ports. conv1 in
// raster order (conv3x3_tiled, a pixel in chunks), fed from a strip buffer (pixel_buffer):
// no patch buffer.

constexpr int PATCH_SIZE = 32;            // patches are 32 x 32 pixels
constexpr int CUBE_BANDS = 120;           // bands per pixel in the raw cube
constexpr int MODEL_BANDS = 110;          // bands the model uses (raw 8-117)
constexpr int BASE_CHANNELS = 6;
constexpr int CLASSES = 3;
constexpr int READ_PORTS = 4;
constexpr int BANDS_PER_WORD = 4;         // one 64-bit word
constexpr int CONV1_BANDS_PER_CYCLE = 12;
// conv1: a pixel's 110 bands in 10 chunks of 12 (the last padded with zero weights), all 6
// outputs a step: 10 x (1 024 + 33) = 10 570 cycles a patch, its pace (10 273 with the patch buffer).
constexpr int CONV1_CHUNKS = (MODEL_BANDS + CONV1_BANDS_PER_CYCLE - 1) / CONV1_BANDS_PER_CYCLE;
constexpr int CONV2_TO_4_CHANNELS_PER_CYCLE = 2;

constexpr int ZSCORE_FRAC_BITS = 5;       // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 11;  // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 16;       // ... of the scores

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

constexpr int HALF_PATCH = PATCH_SIZE / 2;
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = zscore::FIRST_BAND / BANDS_PER_WORD;
constexpr bool ZSCORE_ON_DSPS = false;
constexpr bool CONV2_TO_4_ON_DSPS = true;   // 54 of the 76 DSPs conv1 leaves

// Chip variables for implementation:
constexpr int DSP_PACK_SHIFT = 16;   // DSP48E1, 25-bit input: two weights 16 bits apart (dsp_pack.hpp)

// The argument names become the kernel's register names: keep them.
void justounetsimple_opt_zynq7030(const cube_word_t* din0, const cube_word_t* din1,
                                  const cube_word_t* din2, const cube_word_t* din3,
                                  hls::burst_maxi<score_word_t> dout, int height, int width);

#endif
