// spectralpointwise1d_opt_tb.cpp  --  testbench for the optimized kernel. The checks are in
// spectralpointwise1d_opt_check.hpp, shared with the Zynq-7030 and Zynq-7020 kernels.
#include "spectralpointwise1d_opt.hpp"

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width, uint32_t& cycles) {
    // All four ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    spectralpointwise1d_opt(din, din, din, din, dout, height, width, cycles);
}

#include "spectralpointwise1d_opt_check.hpp"

int main()
{
    return check_kernel();
}
