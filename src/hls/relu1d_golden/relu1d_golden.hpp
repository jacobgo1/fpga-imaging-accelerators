#ifndef RELU1D_GOLDEN_HLS_HPP
#define RELU1D_GOLDEN_HLS_HPP

#include "../../golden/relu1d_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around relu1d_golden.hpp's relu1d_golden
//    template (elementwise ReLU over the [H][W][CH] picture conv1d_golden uses): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/relu1d_golden/)
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
#define RELU1D_H       4
#define RELU1D_W       8
#define RELU1D_CH      3

void relu1d_golden_top(
    data_t din[RELU1D_H][RELU1D_W][RELU1D_CH],
    data_t dout[RELU1D_H][RELU1D_W][RELU1D_CH]);

#endif // RELU1D_GOLDEN_HLS_HPP
