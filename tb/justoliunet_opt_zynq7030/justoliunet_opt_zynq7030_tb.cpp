// justoliunet_opt_zynq7030_tb.cpp  --  testbench for the Zynq-7030 kernel: the same checks as the
// ZCU104's (justoliunet_opt_check.hpp: bit for bit against the plain-loop integer
// reference, close to the float golden network), so the same answers.
#include "justoliunet_opt_zynq7030.hpp"
#include "../../weights/quantized/justoliunet/justoliunet_int8.hpp"
#include "../../src/hls/justoliunet_opt/justoliunet_opt_formats.hpp"

namespace q = justoliunet_int8;
namespace fmt = justoliunet_formats;

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width, uint32_t& cycles) {
    // hls::burst_maxi takes a plain (not const) pointer.
    justoliunet_opt_zynq7030(din, dout, height, width, cycles);
}

#include "../justoliunet_opt/justoliunet_opt_check.hpp"

int main()
{
    return check_kernel();
}
