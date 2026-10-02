#ifndef C1D_GOLDEN_HLS_HPP
#define C1D_GOLDEN_HLS_HPP

#include "../../golden/conv1d_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around conv1d_golden.hpp's conv1d_golden
//    template (PyTorch's Conv1d, applied pointwise per row (see conv1d_golden.hpp)): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/conv1d_golden/)
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
#define C1D_H          4
#define C1D_W          10
#define C1D_IN_CH      2
#define C1D_OUT_CH     3
#define C1D_K          3

#define C1D_OUT_W      ((C1D_W) - (C1D_K) + 1)

void conv1d_golden_top(
    data_t din[C1D_H][C1D_W][C1D_IN_CH],
    data_t weight[C1D_OUT_CH][C1D_IN_CH][C1D_K],
    data_t bias[C1D_OUT_CH],
    data_t dout[C1D_H][C1D_OUT_W][C1D_OUT_CH]);

#endif // C1D_GOLDEN_HLS_HPP
