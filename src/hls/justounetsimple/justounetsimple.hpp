#ifndef JUSTOUNETSIMPLE_HLS_HPP
#define JUSTOUNETSIMPLE_HLS_HPP

#include "../../golden/justounetsimple_golden.hpp"

// Quantized 2D-JustoUNet on one 32 x 32 patch. The int8 weights and their
// shifts come from weights/quantized/justounetsimple/ and are compiled into the
// kernel as constant arrays (on-chip ROM), so the interface is only the patch
// in and the scores out. Changing the weights means rebuilding the bitstream.
//
// The host cuts the capture into 32 x 32 patches, pads where needed, runs the
// kernel once per patch and stitches the results (tools/patching.py).
// din must already be normalized the way the model was trained.

using data_t = float;

#define JUNETS_H         32
#define JUNETS_W         32
#define JUNETS_IN_CH     110
#define JUNETS_BASE_CH   6
#define JUNETS_OUT_CH    3
#define JUNETS_K         3

void justounetsimple(data_t din[JUNETS_H][JUNETS_W][JUNETS_IN_CH],
                     data_t dout[JUNETS_H][JUNETS_W][JUNETS_OUT_CH]);

#endif // JUSTOUNETSIMPLE_HLS_HPP
