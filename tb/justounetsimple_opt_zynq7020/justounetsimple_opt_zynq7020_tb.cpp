// justounetsimple_opt_zynq7020_tb.cpp  --  testbench for the Zynq-7020 version
// of the optimized kernel: the checks of justounetsimple_opt, the same answers.
#include "justounetsimple_opt_zynq7020.hpp"

static void run_kernel(jopt_raw_t *din, jopt_out_t *dout, int height, int width) {
    justounetsimple_opt_zynq7020(din, din, dout, height, width);    // both ports read the cube
}

#include "../justounetsimple_opt/justounetsimple_opt_check.hpp"
