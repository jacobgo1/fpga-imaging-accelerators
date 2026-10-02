#ifndef RELU_GOLDEN_HLS_HPP
#define RELU_GOLDEN_HLS_HPP

#include "../../golden/relu_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around relu_golden.hpp's relu_golden
//    template (PyTorch's elementwise ReLU): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/relu_golden/)
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
#define RELU_H         8
#define RELU_W         8
#define RELU_CH        4

void relu_golden_top(
    data_t din[RELU_H][RELU_W][RELU_CH],
    data_t dout[RELU_H][RELU_W][RELU_CH]);

#endif // RELU_GOLDEN_HLS_HPP
