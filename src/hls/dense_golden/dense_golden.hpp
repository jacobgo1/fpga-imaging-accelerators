#ifndef DENSE_GOLDEN_HLS_HPP
#define DENSE_GOLDEN_HLS_HPP

#include "../../golden/dense_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around dense_golden.hpp's dense_golden
//    template (PyTorch's nn.Linear, applied pointwise per row): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/dense_golden/)
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
#define DENSE_H        4
#define DENSE_IN       12
#define DENSE_OUT      5

void dense_golden_top(
    data_t din[DENSE_H][DENSE_IN],
    data_t weight[DENSE_OUT][DENSE_IN],
    data_t bias[DENSE_OUT],
    data_t dout[DENSE_H][DENSE_OUT]);

#endif // DENSE_GOLDEN_HLS_HPP
