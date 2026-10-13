#pragma once

#include <cstddef>
#include <cstdint>

// The data formats every justounetsimple_opt kernel shares (justounetsimple_opt
// for the ZCU104, justounetsimple_opt_zynq7030 for HYPSO-2): the raw cube in,
// [height][width][120] uint16, and [height][width][4] int32 out (3 scores *
// 2^16, then the class). Kept here rather than taken from a kernel's header so
// that one driver serves both kernels (and named so as not to clash with
// theirs in the sim build); what differs between them (how many
// read ports, where the registers are) comes from the register header.
constexpr uint32_t PATCH = 32;                               // patch height and width
constexpr size_t CUBE_BANDS = 120;
constexpr size_t IN_PIXEL = CUBE_BANDS * sizeof(uint16_t);    // 240 bytes
constexpr size_t OUT_PIXEL = 4 * sizeof(int32_t);            // 16 bytes
