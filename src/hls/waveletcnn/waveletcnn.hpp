#ifndef WAVELETCNN_HLS_HPP
#define WAVELETCNN_HLS_HPP

#include "../../golden/waveletcnn_golden.hpp"

// Quantized WaveletCNN on one 32 x 32 patch. The int8 weights and their shifts
// come from weights/quantized/waveletcnn/ and are compiled into the kernel as
// constant arrays (on-chip ROM), so the interface is only the patch in and the
// scores out. Changing the weights means rebuilding the bitstream.
//
// The host cuts the capture into 32 x 32 patches, pads where needed, runs the
// kernel once per patch and stitches the results (tools/patching.py).
// din must already be normalized the way the model was trained.

using data_t = float;

#define WCNN_H       32
#define WCNN_W       32
#define WCNN_IN_CH   110
#define WCNN_DWT_LEN 8
#define WCNN_SUB_CH  ((WCNN_IN_CH + WCNN_DWT_LEN - 1) / 2)
#define WCNN_CAT_CH  (WCNN_IN_CH + 2 * WCNN_SUB_CH)
#define WCNN_MID_CH  12
#define WCNN_OUT_CH  3

void waveletcnn(data_t din[WCNN_H][WCNN_W][WCNN_IN_CH],
                data_t dout[WCNN_H][WCNN_W][WCNN_OUT_CH]);

#endif // WAVELETCNN_HLS_HPP
