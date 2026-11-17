// Testbench of justoliunet_simple: the same checks as justoliunet_opt (justoliunet_opt_check.hpp),
// since it must give the same answers bit for bit.
#include "justoliunet_simple.hpp"
#include "../../weights/quantized/justoliunet/justoliunet_int8.hpp"
#include "../../src/models/1D-Justo-LiuNet/justoliunet_opt/justoliunet_opt_formats.hpp"

namespace q = justoliunet_int8;
namespace fmt = justoliunet_formats;

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width, uint32_t& cycles) {
    cycles = 0;   // no cycle counter: the board's host timer times it
    justoliunet_simple(din, dout, height, width);
}

#include "../justoliunet_opt/justoliunet_opt_check.hpp"

int main()
{
    return check_kernel();
}
