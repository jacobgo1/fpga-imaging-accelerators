#ifndef SPECTRALPOINTWISE1D_HLS_HPP
#define SPECTRALPOINTWISE1D_HLS_HPP

#include "../../golden/spectralpointwise1d_golden.hpp"

// Golden 1D-SpectralPointwise: n pixels a start, each pixel's 110 z-scored bands
// in, its 3 scores out. Plain float loops, with the int8 weights compiled in.

using data_t = float;

// din: [n][110] float. dout: [n][3] float (0 cloud, 1 land, 2 sea).
void spectralpointwise1d(const data_t *din, data_t *dout, int n);

#endif
