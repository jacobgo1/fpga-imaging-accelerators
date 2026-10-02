#include "relu1d_golden.hpp"

void relu1d_golden_top(
    data_t din[RELU1D_H][RELU1D_W][RELU1D_CH],
    data_t dout[RELU1D_H][RELU1D_W][RELU1D_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    relu1d_golden<data_t, RELU1D_H, RELU1D_W, RELU1D_CH>(din, dout);
}
