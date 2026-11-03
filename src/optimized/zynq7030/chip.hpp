#pragma once

// The Zynq-7030 (HYPSO-2), what the modules need to know of it: the 7020's
// DSP48E1 and no UltraRAM, so the same as zynq7020/chip.hpp.
// DSP48E1, a 25-bit input: two int8 weights 16 bits apart, groups of 2 products
// (conv3x3_2mul.hpp).
constexpr int DSP_PACK_SHIFT = 16, DSP_PACK_GROUP = 2;
// No UltraRAM: the patch buffer is block RAM.
constexpr bool HAS_URAM = false;
