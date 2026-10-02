#include "flatten_golden.hpp"

void flatten_golden_top(
    data_t din[FLAT_H][FLAT_W][FLAT_CH],
    data_t dout[FLAT_H][FLAT_OUT])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    flatten_golden<data_t, FLAT_H, FLAT_W, FLAT_CH>(din, dout);
}
