// sp_unet_large_opt_tb.cpp  --  testbench for the optimized SpecPW-UNet, large. The
// checks are in sp_unet_small_opt/sp_unet_opt_check.hpp, shared by the three sizes.
#include "sp_unet_large_opt.hpp"
// The golden kernel: the float reference, and the weights as namespace q (floats).
#include "../../src/hls/sp_unet_large/sp_unet_large.cpp"
#include <cstdio>

static void run_kernel(spu_raw_t *din, spu_out_t *dout, int height, int width) {
    // All four ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    // cycles counts only in the RTL (cosim): 0 here, in C simulation.
    uint32_t cycles = 0;
    sp_unet_large_opt(din, din, din, din, dout, height, width, cycles);
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
