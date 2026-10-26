// sp_unet_large_tb.cpp  --  testbench for the quantized SpecPW-UNet, large. The
// checks are in sp_unet_check.hpp, shared by the three sizes.
#include "sp_unet_large.hpp"
#include "../../weights/quantized/sp_unet_large/sp_unet_large_int8.hpp"

namespace q = sp_unet_large_int8;

static void run_kernel(const float *din, float *dout, int n) {
    sp_unet_large(din, dout, n);
}

#include "../sp_unet_small/sp_unet_check.hpp"

int main()
{
    return check_kernel();
}
