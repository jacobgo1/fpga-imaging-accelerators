#ifndef WAVELETCNN_OPT_ZYNQ7030_HLS_HPP
#define WAVELETCNN_OPT_ZYNQ7030_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// waveletcnn_opt for the Zynq-7030 (--board zynq7030): the same answers, four
// 64-bit read ports (the writer shares HP3), one conv output channel a cycle.

constexpr int PATCH_SIZE = 32;
constexpr int CUBE_BANDS = 120;
constexpr int MODEL_BANDS = 110;                // raw bands 8-117
constexpr int DWT_TAPS = 8;                     // db4
constexpr int SUBBAND_LENGTH = (MODEL_BANDS + DWT_TAPS - 1) / 2;
constexpr int WAVELET_CHANNELS = MODEL_BANDS + 2 * SUBBAND_LENGTH;
constexpr int MID_CHANNELS = 12;
constexpr int CLASSES = 3;
constexpr int READ_PORTS = 4;
constexpr int BANDS_PER_WORD = 4;
constexpr int DWT_STEPS_PER_CYCLE = 4;
constexpr int DWT_BANDS_PER_CYCLE = 2 * DWT_STEPS_PER_CYCLE;
constexpr int DWT_CYCLES_PER_PIXEL = (SUBBAND_LENGTH + DWT_STEPS_PER_CYCLE - 1) / DWT_STEPS_PER_CYCLE;

constexpr int ZSCORE_FRAC_BITS = 5;             // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 9;         // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 15;             // ... of the scores

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// The argument names become the kernel's register names: keep them.
void waveletcnn_opt_zynq7030(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                        hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                        hls::burst_maxi<score_word_t> dout, int height, int width);

#endif
