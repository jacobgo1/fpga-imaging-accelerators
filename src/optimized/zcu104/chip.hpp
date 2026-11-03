#pragma once

// The ZCU104's ZU7EV, what the modules need to know of it.
// DSP48E2, a 27-bit input: two int8 weights 18 bits apart, groups of 8 products
// (conv3x3_2mul.hpp).
constexpr int DSP_PACK_SHIFT = 18, DSP_PACK_GROUP = 8;
// UltraRAM for the patch buffer.
constexpr bool HAS_URAM = true;
