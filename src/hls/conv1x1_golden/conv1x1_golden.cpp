#include "conv1x1_golden.hpp"

void conv1x1_golden_top(
    data_t din[C1X1_H][C1X1_W][C1X1_IN_CH],
    data_t weight[C1X1_OUT_CH][C1X1_IN_CH],
    data_t bias[C1X1_OUT_CH],
    data_t dout[C1X1_H][C1X1_W][C1X1_OUT_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=weight
    #pragma HLS INTERFACE mode=bram port=bias
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    conv1x1_golden<data_t, C1X1_H, C1X1_W, C1X1_IN_CH, C1X1_OUT_CH>(din, weight, bias, dout);
}
