#ifndef SPECTRALPOINTWISE1D_OPT_ZYNQ7020_HLS_HPP
#define SPECTRALPOINTWISE1D_OPT_ZYNQ7020_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// spectralpointwise1d_opt for the Zynq-7020 (--board zedboard): the same lanes,
// two of them on 64-bit ports, a pixel every 15 cycles. convsp1 and the z-score on
// DSPs (200 of 220), convsp2 and convsp3 in LUTs.

constexpr int CUBE_BANDS = 120;
constexpr int BANDS_PER_WORD = 4;
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = 2;                            // raw bands 0-7 are dropped
constexpr int CLASSES = 3;
constexpr int LANES = 2;                                      // read ports, each with its own network
constexpr int BLOCK_PIXELS = 8;                               // pixels a burst

constexpr bool ZSCORE_ON_DSPS = true;
constexpr bool CONVSP1_ON_DSPS = true;
constexpr int CONVSP2_OUT_PER_CYCLE = 1;   constexpr bool CONVSP2_ON_DSPS = false;
constexpr int CONVSP3_OUT_PER_CYCLE = 1;   constexpr bool CONVSP3_ON_DSPS = false;

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// The argument names become the kernel's register names: keep them.
void spectralpointwise1d_opt_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                                      hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
