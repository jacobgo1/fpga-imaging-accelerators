#include "maxpool1d_golden.hpp"

void maxpool1d_golden_top(
    data_t din[MP1D_H][MP1D_W][MP1D_CH],
    data_t dout[MP1D_H][MP1D_OUT_W][MP1D_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    maxpool1d_golden<data_t, MP1D_H, MP1D_W, MP1D_CH, MP1D_POOL>(din, dout);
}
