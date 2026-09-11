#include "matmul.hpp"

void matmul(const data_t a[MAT_SIZE][MAT_SIZE],
            const data_t b[MAT_SIZE][MAT_SIZE],
            result_t c[MAT_SIZE][MAT_SIZE]) {
#pragma HLS INTERFACE mode=bram port=a
#pragma HLS INTERFACE mode=bram port=b
#pragma HLS INTERFACE mode=bram port=c
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    for (int row = 0; row < MAT_SIZE; ++row) {
        for (int col = 0; col < MAT_SIZE; ++col) {
            result_t sum = 0;
        product_loop:
            for (int k = 0; k < MAT_SIZE; ++k)
                sum += static_cast<result_t>(a[row][k]) * b[k][col];
            c[row][col] = sum;
        }
    }
}
