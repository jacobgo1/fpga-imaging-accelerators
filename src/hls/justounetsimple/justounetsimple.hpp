#ifndef JUSTOUNETSIMPLE_HLS_HPP
#define JUSTOUNETSIMPLE_HLS_HPP

#include "../../golden/justounetsimple_golden.hpp"

// Golden 2D-JustoUNet: one z-scored 32 x 32 patch in, 3 scores per pixel out.
// Plain float loops, with the int8 weights compiled in.

using data_t = float;

constexpr int GOLDEN_PATCH_SIZE = 32;
constexpr int GOLDEN_BANDS = 110;
constexpr int GOLDEN_BASE_CHANNELS = 6;
constexpr int GOLDEN_CLASSES = 3;
constexpr int GOLDEN_KERNEL_SIZE = 3;

void justounetsimple(data_t din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS],
                     data_t dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES]);

#endif
