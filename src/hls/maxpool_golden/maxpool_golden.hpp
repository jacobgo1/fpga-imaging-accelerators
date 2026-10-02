#ifndef MP_GOLDEN_HLS_HPP
#define MP_GOLDEN_HLS_HPP

#include "../../golden/maxpool_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around maxpool_golden.hpp's maxpool_golden
//    template (PyTorch's MaxPool2d): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/maxpool_golden/)
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
#define MP_H           8
#define MP_W           8
#define MP_CH          4
#define MP_POOL        2

#define MP_OUT_H       ((MP_H) / 2)
#define MP_OUT_W       ((MP_W) / 2)

void maxpool_golden_top(
    data_t din[MP_H][MP_W][MP_CH],
    data_t dout[MP_OUT_H][MP_OUT_W][MP_CH]);

#endif // MP_GOLDEN_HLS_HPP
