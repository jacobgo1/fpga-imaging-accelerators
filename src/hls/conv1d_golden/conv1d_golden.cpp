#include "conv1d_golden.hpp"

void conv1d_golden_top(
    data_t din[C1D_H][C1D_W][C1D_IN_CH],
    data_t weight[C1D_OUT_CH][C1D_IN_CH][C1D_K],
    data_t bias[C1D_OUT_CH],
    data_t dout[C1D_H][C1D_OUT_W][C1D_OUT_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=weight
    #pragma HLS INTERFACE mode=bram port=bias
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    conv1d_golden<data_t, C1D_H, C1D_W, C1D_IN_CH, C1D_OUT_CH, C1D_K>(din, weight, bias, dout);
}
