#include "add_golden.hpp"

void add_golden_top(
    data_t a[ADD_H][ADD_W][ADD_CH],
    data_t b[ADD_H][ADD_W][ADD_CH],
    data_t dout[ADD_H][ADD_W][ADD_CH])
{
    #pragma HLS INTERFACE mode=bram port=a
    #pragma HLS INTERFACE mode=bram port=b
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    add_golden<data_t, ADD_H, ADD_W, ADD_CH>(a, b, dout);
}
