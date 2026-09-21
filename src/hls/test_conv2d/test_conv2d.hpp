#ifndef TESTCONV
#define TESTCONV
#include <cstdint>

using data_t = std::int16_t;
using result_t = std::int64_t;

void test_conv2d(const data_t in[3][28][28],
            const data_t weight[3][3][3][3],
            result_t out[3][26][26]);


#endif