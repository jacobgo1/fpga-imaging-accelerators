// justoliunet_tb.cpp  --  testbench for the quantized 1D-Justo-LiuNet kernel.
// The checks are in justoliunet_check.hpp, shared with justoliunet_bn.
#include "justoliunet.hpp"
#include "../../weights/quantized/justoliunet/justoliunet_int8.hpp"

namespace q = justoliunet_int8;

static void run_kernel(const float *din, float *dout, int n) {
    justoliunet(din, dout, n);
}

#include "justoliunet_check.hpp"

int main()
{
    return check_kernel();
}
