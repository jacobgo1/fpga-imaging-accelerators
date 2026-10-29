#ifndef SPECTRALPOINTWISE1D_OPT_HLS_HPP
#define SPECTRALPOINTWISE1D_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The quantized 1D-SpectralPointwise on a whole raw capture. The network is small
// and the kernel DDR-bound, so it is copied once per read port (a lane), each lane
// classifying its reader's blocks; the store puts the scores back in pixel order.
// convsp1 runs on the raw words as they arrive (zscore_pointwise.hpp). Four lanes,
// a pixel every 3.75 cycles. Design and numbers: optimizations_implemented/.

constexpr int CUBE_BANDS = 120;
constexpr int BANDS_PER_WORD = 8;
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = 1;                            // raw bands 0-7 are dropped
constexpr int CLASSES = 3;
constexpr int LANES = 4;                                      // read ports, each with its own network
constexpr int BLOCK_PIXELS = 16;                               // pixels a burst

constexpr bool ZSCORE_ON_DSPS = true;
constexpr bool CONVSP1_ON_DSPS = true;
constexpr int CONVSP2_OUT_PER_CYCLE = 1;   constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_OUT_PER_CYCLE = 1;   constexpr bool CONVSP3_ON_DSPS = false;

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// The argument names become the kernel's register names: keep them.
void spectralpointwise1d_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                             hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                             hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles);

#endif
