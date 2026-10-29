// Testbench of justounetsimple_opt; the checks are in justounetsimple_opt_check.hpp.
#include "justounetsimple_opt.hpp"
#include <cstdio>

static void run_kernel(cube_word_t *cube, score_word_t *scores, int height, int width) {
    uint32_t cycles = 0;   // only counts in co-simulation
    justounetsimple_opt(cube, cube, cube, cube, scores, height, width, cycles);
    std::printf("cycles register: %u\n", (unsigned)cycles);
}

#include "justounetsimple_opt_check.hpp"
