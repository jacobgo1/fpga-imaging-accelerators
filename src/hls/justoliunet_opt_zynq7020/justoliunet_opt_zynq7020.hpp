#ifndef JUSTOLIUNET_OPT_ZYNQ7020_HLS_HPP
#define JUSTOLIUNET_OPT_ZYNQ7020_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// justoliunet_opt for the Zynq-7020 (--board zedboard): the same answers, bit for bit.
// 220 DSPs cannot compute positions side by side, so each layer takes one output
// position at a time, folded over steps, and its input arrives one position at a
// time (conv1d_stream.hpp). One pixel every 108 cycles, one read port.

constexpr int CUBE_BANDS = 120;
constexpr int BANDS_PER_WORD = 4;                                  // one 64-bit word
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = 2;                                 // raw bands 0-7 are dropped
constexpr int KEPT_BANDS = (WORDS_PER_PIXEL - FIRST_KEPT_WORD) * BANDS_PER_WORD;   // 110 and 111 are zero
constexpr int KERNEL_SIZE = 6;
constexpr int CLASSES = 3;
constexpr int READ_PORTS = 1;
constexpr int BLOCK_PIXELS = 8;                                    // pixels a burst

constexpr int CONV1_POSITIONS_PER_CYCLE = 1;
constexpr int CONV2_OUT_PER_CYCLE = 6, CONV2_IN_PER_CYCLE = 6;   constexpr bool CONV2_PACKED = true;
constexpr int CONV3_OUT_PER_CYCLE = 6, CONV3_IN_PER_CYCLE = 6;   constexpr bool CONV3_PACKED = true;
constexpr int CONV4_OUT_PER_CYCLE = 2, CONV4_IN_PER_CYCLE = 9;   constexpr bool CONV4_PACKED = false;
constexpr int DENSE_FEATURES_PER_CYCLE = 1;
constexpr bool ZSCORE_ON_DSPS = false;

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// The argument names become the kernel's register names: keep them.
void justoliunet_opt_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                              int height, int width, uint32_t& cycles);

#endif
