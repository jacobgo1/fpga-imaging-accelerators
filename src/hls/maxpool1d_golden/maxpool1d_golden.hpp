#ifndef MP1D_GOLDEN_HLS_HPP
#define MP1D_GOLDEN_HLS_HPP

#include "../../golden/maxpool1d_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around maxpool1d_golden.hpp's maxpool1d_golden
//    template (PyTorch's MaxPool1d with stride == kernel size, applied pointwise per row): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/maxpool1d_golden/)
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
#define MP1D_H         4
#define MP1D_W         8
#define MP1D_CH        3
#define MP1D_POOL      2

#define MP1D_OUT_W     ((MP1D_W) / (MP1D_POOL))

void maxpool1d_golden_top(
    data_t din[MP1D_H][MP1D_W][MP1D_CH],
    data_t dout[MP1D_H][MP1D_OUT_W][MP1D_CH]);

#endif // MP1D_GOLDEN_HLS_HPP
