#pragma once

// 2^-frac, for the quantized networks' scale arguments (weights/quantized/:
// real value = q * 2^-frac). constexpr: evaluated by the compiler, so no
// conversion is left for the hardware.
constexpr float pow2_neg(int frac) {
    return 1.0f / (float)(1L << frac);
}
