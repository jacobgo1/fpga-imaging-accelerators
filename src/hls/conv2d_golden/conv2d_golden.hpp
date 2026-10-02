#ifndef C2D_GOLDEN_HLS_HPP
#define C2D_GOLDEN_HLS_HPP

#include "../../golden/conv2d_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around conv2d_golden.hpp's conv2d_golden
//    template (PyTorch's Conv2d, valid (unpadded) convolution): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/conv2d_golden/)
//    calls the same golden template directly as its expected reference,
//    so a PASS there proves wiring this through interface pragmas didn't
//    change what gets computed.
// =====================================================================

// =====================================================================
// 2. Data type -- float, not fixed-point: numerically identical to the
//    golden reference, same reasoning as justoliunet.hpp's data_t.
// =====================================================================
using data_t = float;

// =====================================================================
// 3. Geometry -- concrete sizes to synthesize/cosim against. #define,
//    not constexpr, so every value here is also valid pragma-argument
//    text if a later variant needs that (see conv2d.hpp part 2).
// =====================================================================
#define C2D_H          10
#define C2D_W          10
#define C2D_IN_CH      3
#define C2D_OUT_CH     4
#define C2D_K          3

#define C2D_OUT_H      ((C2D_H) - (C2D_K) + 1)
#define C2D_OUT_W      ((C2D_W) - (C2D_K) + 1)

void conv2d_golden_top(
    data_t din[C2D_H][C2D_W][C2D_IN_CH],
    data_t weight[C2D_OUT_CH][C2D_IN_CH][C2D_K][C2D_K],
    data_t bias[C2D_OUT_CH],
    data_t dout[C2D_OUT_H][C2D_OUT_W][C2D_OUT_CH]);

#endif // C2D_GOLDEN_HLS_HPP
