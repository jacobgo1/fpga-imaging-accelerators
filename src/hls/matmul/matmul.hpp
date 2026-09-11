#ifndef IMAGING_MATMUL_HPP
#define IMAGING_MATMUL_HPP
#include <cstdint>

constexpr int MAT_SIZE = 8;
using data_t = std::int16_t;
// The worst-case sum of 8 signed 16-bit products needs 35 bits, so int64_t is
// safe but wider than the hardware needs; ap_int<35> is the obvious first
// area optimization once you start comparing synthesis results.
using result_t = std::int64_t;

void matmul(const data_t a[MAT_SIZE][MAT_SIZE],
            const data_t b[MAT_SIZE][MAT_SIZE],
            result_t c[MAT_SIZE][MAT_SIZE]);
#endif
