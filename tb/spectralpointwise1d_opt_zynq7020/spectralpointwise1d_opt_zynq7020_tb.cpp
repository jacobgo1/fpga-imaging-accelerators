// spectralpointwise1d_opt_zynq7020_tb.cpp  --  testbench for the Zynq-7020 kernel: the same checks
// as the ZCU104's (spectralpointwise1d_opt_check.hpp: bit for bit against the plain-loop integer
// reference, close to the float golden network), so the same answers.
#include "spectralpointwise1d_opt_zynq7020.hpp"

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width, uint32_t& cycles) {
    // Both ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    spectralpointwise1d_opt_zynq7020(din, din, dout, height, width, cycles);
}

#include "../spectralpointwise1d_opt/spectralpointwise1d_opt_check.hpp"

int main()
{
    return check_kernel();
}
