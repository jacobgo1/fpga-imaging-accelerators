#include "nearest_neighbor_golden.hpp"

void nearest_neighbor_golden_top(
    data_t din[NNUP_HIN][NNUP_WIN][NNUP_CH],
    data_t dout[NNUP_HOUT][NNUP_WOUT][NNUP_CH])
{
    #pragma HLS INTERFACE mode=bram port=din
    #pragma HLS INTERFACE mode=bram port=dout
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    nearest_neighbor_golden<data_t, NNUP_HIN, NNUP_WIN, NNUP_CH, NNUP_FACTOR>(din, dout);
}
