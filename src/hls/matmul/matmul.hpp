#ifndef IMAGING_MATMUL_HPP
#define IMAGING_MATMUL_HPP

#include <cstdint>

// A minimal example kernel: an 8 x 8 matrix product, used by the framework tests.

constexpr int MAT_SIZE = 8;
using data_t = std::int16_t;
using result_t = std::int64_t;   // a sum of 8 int16 products needs 35 bits

void matmul(const data_t a[MAT_SIZE][MAT_SIZE],
            const data_t b[MAT_SIZE][MAT_SIZE],
            result_t c[MAT_SIZE][MAT_SIZE]);

#endif
