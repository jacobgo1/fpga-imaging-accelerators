#include "padding_golden.hpp"

void padding_golden_top(
    data_t din[PAD_H][PAD_W][PAD_CH],
    data_t dout[PAD_OUT_H][PAD_OUT_W][PAD_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    padding_golden<data_t, PAD_H, PAD_W, PAD_CH, PAD_PAD>(din, dout);
}
