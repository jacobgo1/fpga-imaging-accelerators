#pragma once

// The Zynq-7020 (PYNQ-Z2, ZedBoard), what the modules need to know of it.
// DSP48E1, a 25-bit input: two int8 weights 16 bits apart, groups of 2 products
// (conv3x3_2mul.hpp).
constexpr int DSP_PACK_SHIFT = 16, DSP_PACK_GROUP = 2;
// No UltraRAM: the patch buffer is block RAM.
constexpr bool HAS_URAM = false;
