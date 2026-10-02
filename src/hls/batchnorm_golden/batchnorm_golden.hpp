#ifndef BN_GOLDEN_HLS_HPP
#define BN_GOLDEN_HLS_HPP

#include "../../golden/batchnorm_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around batchnorm_golden.hpp's batchnorm_golden
//    template (PyTorch's BatchNorm2d in inference mode (running stats, no batch stats)): fixes the template parameters to concrete
//    geometry below and adds the interface pragmas synthesis needs. The
//    golden template still owns 100% of the arithmetic -- this file adds
//    no logic of its own, same relationship as justoliunet.cpp has to
//    justoliunet_golden.hpp. The matching testbench (tb/batchnorm_golden/)
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
#define BN_H           8
#define BN_W           8
#define BN_CH          4

void batchnorm_golden_top(
    data_t din[BN_H][BN_W][BN_CH],
    data_t weight[BN_CH],
    data_t bias[BN_CH],
    data_t running_mean[BN_CH],
    data_t running_var[BN_CH],
    data_t eps,
    data_t dout[BN_H][BN_W][BN_CH]);

#endif // BN_GOLDEN_HLS_HPP
