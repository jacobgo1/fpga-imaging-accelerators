#include "batchnorm_golden.hpp"

void batchnorm_golden_top(
    data_t din[BN_H][BN_W][BN_CH],
    data_t weight[BN_CH],
    data_t bias[BN_CH],
    data_t running_mean[BN_CH],
    data_t running_var[BN_CH],
    data_t eps,
    data_t dout[BN_H][BN_W][BN_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=weight
    #pragma HLS INTERFACE mode=bram port=bias
    #pragma HLS INTERFACE mode=bram port=running_mean
    #pragma HLS INTERFACE mode=bram port=running_var
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=eps bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    batchnorm_golden<data_t, BN_H, BN_W, BN_CH>(din, weight, bias, running_mean, running_var, eps, dout);
}
