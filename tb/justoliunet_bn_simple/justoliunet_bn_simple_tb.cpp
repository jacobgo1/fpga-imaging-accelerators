// Testbench of justoliunet_bn_simple: the same checks as justoliunet_bn_opt
// (justoliunet_opt_check.hpp), since it must give the same answers bit for bit.
#include "justoliunet_bn_simple.hpp"
#include "../../weights/quantized/justoliunet_bn/justoliunet_bn_int8.hpp"
#include "../../src/models/1D-Justo-LiuNet/justoliunet_opt/justoliunet_opt_formats.hpp"

namespace q = justoliunet_bn_int8;
namespace fmt = justoliunet_bn_formats;

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width, uint32_t& cycles) {
    cycles = 0;   // no cycle counter: the board's host timer times it
    justoliunet_bn_simple(din, dout, height, width);
}

#include "../justoliunet_opt/justoliunet_opt_check.hpp"

int main()
{
    return check_kernel();
}
