// justounetsimple_opt_zynq7030_tb.cpp  --  testbench for the Zynq-7030 version
// of the optimized kernel: the checks of justounetsimple_opt, the same answers.
#include "justounetsimple_opt_zynq7030.hpp"

static void run_kernel(const jopt_raw_t *din, int32_t *dout, int height, int width) {
    justounetsimple_opt_zynq7030(din, din, dout, height, width);    // both ports read the cube
}

#include "../justounetsimple_opt/justounetsimple_opt_check.hpp"
