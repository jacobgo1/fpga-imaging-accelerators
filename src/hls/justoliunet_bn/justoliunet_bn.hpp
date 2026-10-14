#ifndef JUSTOLIUNET_BN_HLS_HPP
#define JUSTOLIUNET_BN_HLS_HPP

#include "../../golden/justoliunet_golden.hpp"

// Quantized 1D-Justo-LiuNet with BatchNorm (src/golden/justoliunet_golden.hpp) on n pixels per start:
// for each pixel, its 110-band spectrum in, its 3 class scores out. The int8
// weights and their shifts come from weights/quantized/justoliunet_bn/ and are
// compiled into the kernel as constant arrays (on-chip ROM), so the interface
// is only the spectra in and the scores out. Changing the weights means
// rebuilding the bitstream.
// The trained model has a BatchNorm after conv1 (conv1 -> bn1 -> ReLU);
// tools/quantize_weights.py folded it into conv1, so the network is the same as
// justoliunet's and only the weights differ.
//
// A 1D network: each pixel's convolutions run along its bands, so the pixels
// are independent and the host may send any number of them, in any order.
// din must already be preprocessed the way the model was trained: the 110 kept
// bands (raw 8-117), z-scored with mu_sd.txt (tools/justounetsimple_model.py,
// read_preprocessing).
//
// din:  [n][110] float, pixel by pixel.
// dout: [n][3] float, the scores (0 cloud, 1 land, 2 sea; highest = class).

using data_t = float;

void justoliunet_bn(const data_t *din, data_t *dout, int n);

#endif // JUSTOLIUNET_BN_HLS_HPP
