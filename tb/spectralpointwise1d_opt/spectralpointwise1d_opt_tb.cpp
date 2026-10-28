// spectralpointwise1d_opt_tb.cpp  --  testbench for the optimized kernel. The checks are in
// spectralpointwise1d_opt_check.hpp, shared with the Zynq-7030 and Zynq-7020 kernels.
#include "spectralpointwise1d_opt.hpp"

static void run_kernel(spw_raw_t *din, spw_out_t *dout, int height, int width, uint32_t& cycles) {
    // All four ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    spectralpointwise1d_opt(din, din, din, din, dout, height, width, cycles);
}

#include "spectralpointwise1d_opt_check.hpp"

int main()
{
    return check_kernel();
}
