#include "relu_golden.hpp"

void relu_golden_top(
    data_t din[RELU_H][RELU_W][RELU_CH],
    data_t dout[RELU_H][RELU_W][RELU_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    relu_golden<data_t, RELU_H, RELU_W, RELU_CH>(din, dout);
}
