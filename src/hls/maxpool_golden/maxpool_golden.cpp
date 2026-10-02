#include "maxpool_golden.hpp"

void maxpool_golden_top(
    data_t din[MP_H][MP_W][MP_CH],
    data_t dout[MP_OUT_H][MP_OUT_W][MP_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    maxpool_golden<data_t, MP_H, MP_W, MP_CH, MP_POOL>(din, dout);
}
