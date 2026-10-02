#ifndef PAD_GOLDEN_HLS_HPP
#define PAD_GOLDEN_HLS_HPP

#include "../../golden/padding_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around padding_golden.hpp's padding_golden
//    template (zero-padding a feature map (the step conv2d_padded_golden uses internally)): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/padding_golden/)
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
#define PAD_H          8
#define PAD_W          8
#define PAD_CH         3
#define PAD_PAD        1

#define PAD_OUT_H      ((PAD_H) + 2*(PAD_PAD))
#define PAD_OUT_W      ((PAD_W) + 2*(PAD_PAD))

void padding_golden_top(
    data_t din[PAD_H][PAD_W][PAD_CH],
    data_t dout[PAD_OUT_H][PAD_OUT_W][PAD_CH]);

#endif // PAD_GOLDEN_HLS_HPP
