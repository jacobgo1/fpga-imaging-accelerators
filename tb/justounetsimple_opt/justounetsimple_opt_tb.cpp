// justounetsimple_opt_tb.cpp  --  testbench for the optimized kernel. The
// checks are in justounetsimple_opt_check.hpp, shared with the Zynq-7030 one.
#include "justounetsimple_opt.hpp"

static void run_kernel(jopt_raw_t *din, jopt_out_t *dout, int height, int width) {
    // All four ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    justounetsimple_opt(din, din, din, din, dout, height, width);
}

#include "justounetsimple_opt_check.hpp"
