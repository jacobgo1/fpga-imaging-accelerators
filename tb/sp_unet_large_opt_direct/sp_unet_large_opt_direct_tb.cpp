// sp_unet_large_opt_direct_tb.cpp  --  testbench for the direct SpecPW-UNet, large:
// the same checks as the Winograd kernels' (sp_unet_small_opt/sp_unet_opt_check.hpp).
#include "sp_unet_large_opt_direct.hpp"
// The golden kernel: the float reference, and the weights as namespace weights (floats).
#include "../../src/hls/sp_unet_large/sp_unet_large.cpp"
#include <cstdio>

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width) {
    // All four ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    // cycles counts only in the RTL (cosim): 0 here, in C simulation.
    uint32_t cycles = 0;
    sp_unet_large_opt_direct(din, din, din, din, dout, height, width, cycles);
    std::printf("cycles register: %u\n", (unsigned)cycles);
}

static void run_golden(const float *din, float *dout, int n) {
    sp_unet_large(din, dout, n);
}

#include "../sp_unet_small_opt/sp_unet_opt_check.hpp"

int main()
{
    return check_kernel();
}
