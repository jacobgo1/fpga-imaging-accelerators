// sp_unet_small_opt_u8_zynq7020_tb.cpp  --  testbench for SpecPW-UNet small with uint8
// activations on the Zynq-7020: the checks of the uint8 kernels
// (sp_unet_small_opt/sp_unet_opt_check.hpp), the same answers as sp_unet_small_opt_u8.
#include "sp_unet_small_opt_u8_zynq7020.hpp"
// The golden kernel: the float reference, and the weights as namespace weights (floats).
#include "../../src/models/SpecPW-UNet/Small/sp_unet_small_golden/sp_unet_small.cpp"
#include <cstdio>

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width) {
    // cycles counts only in the RTL (cosim): 0 here, in C simulation.
    uint32_t cycles = 0;
    sp_unet_small_opt_u8_zynq7020(din, dout, height, width, cycles);
    std::printf("cycles register: %u\n", (unsigned)cycles);
}

static void run_golden(const float *din, float *dout, int n) {
    sp_unet_small(din, dout, n);
}

#include "../sp_unet_small_opt/sp_unet_opt_check.hpp"

int main()
{
    return check_kernel_u8(SPECPW_UNET_FRACS_GOLDEN(act8));
}
