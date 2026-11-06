// sp_unet_large_opt_u8_tb.cpp  --  testbench for SpecPW-UNet large with uint8 activations:
// the checks are in sp_unet_small_opt/sp_unet_opt_check.hpp, shared by the sizes, here
// against the uint8 arithmetic (each layer's fraction bits, saturated at 255).
#include "sp_unet_large_opt_u8.hpp"
// The golden kernel: the float reference, and the weights as namespace weights (floats).
#include "../../src/models/SpecPW-UNet/Large/sp_unet_large_golden/sp_unet_large.cpp"
#include <cstdio>

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width) {
    // All four ports read the cube.
    // cycles counts only in the RTL (cosim): 0 here, in C simulation.
    uint32_t cycles = 0;
    sp_unet_large_opt_u8(din, din, din, din, dout, height, width, cycles);
    std::printf("cycles register: %u\n", (unsigned)cycles);
}

static void run_golden(const float *din, float *dout, int n) {
    sp_unet_large(din, dout, n);
}

#include "../sp_unet_small_opt/sp_unet_opt_check.hpp"

int main()
{
    return check_kernel_u8(SPECPW_UNET_FRACS_GOLDEN(act8));
}
