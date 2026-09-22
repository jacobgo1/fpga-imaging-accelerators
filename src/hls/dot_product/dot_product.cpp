#include "dot_product.h"

void dot_product(float a[N], float b[N], float *result) {
    #pragma HLS ARRAY_PARTITION variable=a cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=b cyclic factor=8 dim=1

    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;

    LOOP_DOT:
    for (int i = 0; i < N; i += 8) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=8
        sum0 += a[i]   * b[i];
        sum1 += a[i+1] * b[i+1];
        sum2 += a[i+2] * b[i+2];
        sum3 += a[i+3] * b[i+3];
        sum4 += a[i+4] * b[i+4];
        sum5 += a[i+5] * b[i+5];
        sum6 += a[i+6] * b[i+6];
        sum7 += a[i+7] * b[i+7];
    }

    *result = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
}