#ifndef NNUP_GOLDEN_HLS_HPP
#define NNUP_GOLDEN_HLS_HPP

#include "../../golden/nearest_neighbor_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around nearest_neighbor_golden.hpp's nearest_neighbor_golden
//    template (nearest-neighbor upsampling (PyTorch's F.interpolate(mode="nearest"))): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/nearest_neighbor_golden/)
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
#define NNUP_HIN       4
#define NNUP_WIN       4
#define NNUP_CH        4
#define NNUP_FACTOR    2

#define NNUP_HOUT      ((NNUP_HIN) * (NNUP_FACTOR))
#define NNUP_WOUT      ((NNUP_WIN) * (NNUP_FACTOR))

void nearest_neighbor_golden_top(
    data_t din[NNUP_HIN][NNUP_WIN][NNUP_CH],
    data_t dout[NNUP_HOUT][NNUP_WOUT][NNUP_CH]);

#endif // NNUP_GOLDEN_HLS_HPP
