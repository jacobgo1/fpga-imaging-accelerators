// justoliunet_opt_tb.cpp  --  testbench for the optimized kernel. The checks are in
// justoliunet_opt_check.hpp, shared by justoliunet_opt and justoliunet_bn_opt.
#include "justoliunet_opt.hpp"
#include "../../weights/quantized/justoliunet/justoliunet_int8.hpp"
#include "../../src/hls/justoliunet_opt/justoliunet_opt_formats.hpp"

namespace q = justoliunet_int8;
namespace fmt = justoliunet_formats;

static void run_kernel(jl_raw_t *din, jl_out_t *dout, int height, int width, uint32_t& cycles) {
    // Both ports read the cube; hls::burst_maxi takes a plain (not const) pointer.
    justoliunet_opt(din, din, dout, height, width, cycles);
}

#include "justoliunet_opt_check.hpp"

int main()
{
    return check_kernel();
}
