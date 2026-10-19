#pragma once

// conv3x3_wide for the Zynq-7030: the 7020's (two multiplies per DSP, packed 16
// bits apart, groups of 2), since both have the 7-series DSP48E1. The 7030 has
// more of them, so its kernel picks a larger P, not another module.
#include "../zynq7020/conv3x3_wide.hpp"
