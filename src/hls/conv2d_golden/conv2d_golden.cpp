#include "conv2d_golden.hpp"

void conv2d_golden_top(
    data_t din[C2D_H][C2D_W][C2D_IN_CH],
    data_t weight[C2D_OUT_CH][C2D_IN_CH][C2D_K][C2D_K],
    data_t bias[C2D_OUT_CH],
    data_t dout[C2D_OUT_H][C2D_OUT_W][C2D_OUT_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=weight
    #pragma HLS INTERFACE mode=bram port=bias
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    conv2d_golden<data_t, C2D_H, C2D_W, C2D_IN_CH, C2D_OUT_CH, C2D_K>(din, weight, bias, dout);
}
