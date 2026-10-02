#include "dense_golden.hpp"

void dense_golden_top(
    data_t din[DENSE_H][DENSE_IN],
    data_t weight[DENSE_OUT][DENSE_IN],
    data_t bias[DENSE_OUT],
    data_t dout[DENSE_H][DENSE_OUT])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=weight
    #pragma HLS INTERFACE mode=bram port=bias
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    dense_golden<data_t, DENSE_H, DENSE_IN, DENSE_OUT>(din, weight, bias, dout);
}
