#ifndef WAVELETCNN_HLS_HPP
#define WAVELETCNN_HLS_HPP

#include "../../golden/waveletcnn_golden.hpp"

// Golden WaveletCNN: one z-scored 32 x 32 patch in, 3 scores per pixel out.
// Plain float loops, with the int8 weights compiled in.

using data_t = float;

constexpr int GOLDEN_PATCH_SIZE = 32;
constexpr int GOLDEN_BANDS = 110;
constexpr int GOLDEN_DWT_TAPS = 8;
constexpr int GOLDEN_WAVELET_CHANNELS = dwt1d_output_channels(GOLDEN_BANDS, GOLDEN_DWT_TAPS);
constexpr int GOLDEN_MID_CHANNELS = 12;
constexpr int GOLDEN_CLASSES = 3;

void waveletcnn(data_t din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS],
                data_t dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES]);

#endif
