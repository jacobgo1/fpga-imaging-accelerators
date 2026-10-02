#ifndef C1X1_GOLDEN_HLS_HPP
#define C1X1_GOLDEN_HLS_HPP

#include "../../golden/conv1x1_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around conv1x1_golden.hpp's conv1x1_golden
//    template (PyTorch's Conv2d(kernel_size=1) -- a per-pixel channel mix): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/conv1x1_golden/)
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
#define C1X1_H         8
#define C1X1_W         8
#define C1X1_IN_CH     4
#define C1X1_OUT_CH    6

void conv1x1_golden_top(
    data_t din[C1X1_H][C1X1_W][C1X1_IN_CH],
    data_t weight[C1X1_OUT_CH][C1X1_IN_CH],
    data_t bias[C1X1_OUT_CH],
    data_t dout[C1X1_H][C1X1_W][C1X1_OUT_CH]);

#endif // C1X1_GOLDEN_HLS_HPP
