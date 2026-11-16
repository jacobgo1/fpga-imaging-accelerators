// Testbench of sp_unet_large_simple: the same checks as sp_unet_large_opt
// (sp_unet_small_opt/sp_unet_opt_check.hpp), since it must give the same answers bit for bit.
#include "sp_unet_large_simple.hpp"
// The golden kernel: the float reference, and the weights as namespace weights (floats).
#include "../../src/models/SpecPW-UNet/Large/sp_unet_large_golden/sp_unet_large.cpp"
#include <cstdio>

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width) {
    sp_unet_large_simple(din, dout, height, width);
}

static void run_golden(const float *din, float *dout, int n) {
    sp_unet_large(din, dout, n);
}

#include "../sp_unet_small_opt/sp_unet_opt_check.hpp"

int main()
{
    return check_kernel();
}
