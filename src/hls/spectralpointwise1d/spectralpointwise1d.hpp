#ifndef SPECTRALPOINTWISE1D_HLS_HPP
#define SPECTRALPOINTWISE1D_HLS_HPP

#include "../../golden/spectralpointwise1d_golden.hpp"

// Quantized 1D-SpectralPointwise (src/golden/spectralpointwise1d_golden.hpp) on
// n pixels per start: for each pixel, its 110-band spectrum in, its 3 class
// scores out. The int8 weights and their shifts come from
// weights/quantized/spectralpointwise1d/ and are compiled into the kernel as
// constant arrays (on-chip ROM), so the interface is only the spectra in and
// the scores out. Changing the weights means rebuilding the bitstream.
// The trained model has a BatchNorm after convsp1; tools/quantize_weights.py
// folded it into convsp1.
//
// A per-pixel network (three 1x1 convs): the pixels are independent and the
// host may send any number of them, in any order. din must already be
// preprocessed the way the model was trained: the 110 kept bands (raw 8-117),
// z-scored with mu_sd.txt (tools/justounetsimple_model.py, read_preprocessing),
// the same input as justoliunet's.
//
// din:  [n][110] float, pixel by pixel.
// dout: [n][3] float, the scores (0 cloud, 1 land, 2 sea; highest = class).

using data_t = float;

void spectralpointwise1d(const data_t *din, data_t *dout, int n);

#endif // SPECTRALPOINTWISE1D_HLS_HPP
