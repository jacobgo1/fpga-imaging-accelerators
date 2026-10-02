#include "matmul.hpp"

void matmul(const data_t a[MAT_SIZE][MAT_SIZE],
            const data_t b[MAT_SIZE][MAT_SIZE],
            result_t c[MAT_SIZE][MAT_SIZE]) {
#pragma HLS INTERFACE mode=s_axilite port=a bundle=control
#pragma HLS INTERFACE mode=s_axilite port=b bundle=control
#pragma HLS INTERFACE mode=s_axilite port=c bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    data_t a_buf[MAT_SIZE][MAT_SIZE];
    data_t b_buf[MAT_SIZE][MAT_SIZE];
#pragma HLS ARRAY_PARTITION variable=a_buf type=complete dim=2  // all k of one row per cycle
#pragma HLS ARRAY_PARTITION variable=b_buf type=complete dim=1  // all k of one column per cycle

copy_in:
    for (int i = 0; i < MAT_SIZE; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
#pragma HLS PIPELINE II=1
            a_buf[i][j] = a[i][j];
            b_buf[i][j] = b[i][j];
        }
    }

compute:
    for (int row = 0; row < MAT_SIZE; ++row) {
        for (int col = 0; col < MAT_SIZE; ++col) {
#pragma HLS PIPELINE II=1
            result_t sum = 0;
        product_loop:
            for (int k = 0; k < MAT_SIZE; ++k) {
#pragma HLS UNROLL
                sum += static_cast<result_t>(a_buf[row][k]) * b_buf[k][col];
            }
            c[row][col] = sum;
        }
    }
}