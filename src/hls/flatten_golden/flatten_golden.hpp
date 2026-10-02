#ifndef FLAT_GOLDEN_HLS_HPP
#define FLAT_GOLDEN_HLS_HPP

#include "../../golden/flatten_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around flatten_golden.hpp's flatten_golden
//    template (PyTorch's torch.flatten(x, 1), applied pointwise per row): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/flatten_golden/)
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
#define FLAT_H         4
#define FLAT_W         4
#define FLAT_CH        3

#define FLAT_OUT       ((FLAT_CH) * (FLAT_W))

void flatten_golden_top(
    data_t din[FLAT_H][FLAT_W][FLAT_CH],
    data_t dout[FLAT_H][FLAT_OUT]);

#endif // FLAT_GOLDEN_HLS_HPP
