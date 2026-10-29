#ifndef JUSTOLIUNET_OPT_HLS_HPP
#define JUSTOLIUNET_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The quantized 1D-Justo-LiuNet on a whole raw capture, one pixel every ~18
// cycles. Each layer computes several output positions at once, sized so all
// take about the same cycles a pixel; the DSPs go to conv2 and conv3 (75 % of
// the work), two multiplies each. Design and numbers: optimizations_implemented/.

constexpr int CUBE_BANDS = 120;
constexpr int BANDS_PER_WORD = 8;                                  // one 128-bit word
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = 1;                                 // raw bands 0-7 are dropped
constexpr int KEPT_BANDS = (WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD;   // 110 and 111 are zero
constexpr int KERNEL_SIZE = 6;
constexpr int CLASSES = 3;
constexpr int READ_PORTS = 2;                                      // alternate blocks
constexpr int BLOCK_PIXELS = 16;                                   // pixels a burst

constexpr int CONV1_POSITIONS_PER_CYCLE = 6;   constexpr bool CONV1_ON_DSPS = false;
constexpr int CONV2_POSITIONS_PER_CYCLE = 3;   constexpr bool CONV2_PACKED = true;
constexpr int CONV3_POSITIONS_PER_CYCLE = 1;   constexpr bool CONV3_PACKED = true;
constexpr int CONV4_POSITIONS_PER_CYCLE = 1;   constexpr bool CONV4_PACKED = false;
constexpr int CONV4_OUT_PER_CYCLE = 6;
constexpr int DENSE_FEATURES_PER_CYCLE = 3;
constexpr bool ZSCORE_ON_DSPS = true;

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// The argument names become the kernel's register names: keep them.
void justoliunet_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                     hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
