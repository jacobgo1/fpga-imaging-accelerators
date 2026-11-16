// Testbench of spectralpointwise1d_simple: the same checks as spectralpointwise1d_opt
// (spectralpointwise1d_opt_check.hpp), since it must give the same answers bit for bit.
#include "spectralpointwise1d_simple.hpp"

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width, uint32_t& cycles) {
    cycles = 0;   // no cycle counter: the board's host timer times it
    spectralpointwise1d_simple(din, dout, height, width);
}

#include "../spectralpointwise1d_opt/spectralpointwise1d_opt_check.hpp"

int main()
{
    return check_kernel();
}
