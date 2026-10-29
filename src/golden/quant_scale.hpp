#pragma once

// The quantized weights are q * 2^-frac: this is that 2^-frac.
constexpr float frac_to_scale(int frac) {
    return 1.0f / (float)(1L << frac);
}
