// justounetsimple_opt_tb.cpp  --  testbench for the optimized kernel. The
// checks are in justounetsimple_opt_check.hpp, shared with the Zynq-7030 one.
#include "justounetsimple_opt.hpp"

static void run_kernel(const jopt_raw_t *din, int32_t *dout, int height, int width) {
    justounetsimple_opt(din, din, din, din, dout, height, width);   // all four ports read the cube
}

#include "justounetsimple_opt_check.hpp"
