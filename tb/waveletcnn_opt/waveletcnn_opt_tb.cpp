// waveletcnn_opt_tb.cpp  --  testbench for the optimized WaveletCNN kernel.
// The checks are in waveletcnn_opt_check.hpp, shared with the Zynq-7000 ones.
#include "waveletcnn_opt.hpp"
#include <cstdio>

static void run_kernel(wopt_raw_t *din, wopt_out_t *dout, int height, int width) {
    uint32_t cycles = 0;
    waveletcnn_opt(din, din, din, din, dout, height, width, cycles);
    std::printf("cycles register: %u\n", (unsigned)cycles);
}

#include "waveletcnn_opt_check.hpp"
