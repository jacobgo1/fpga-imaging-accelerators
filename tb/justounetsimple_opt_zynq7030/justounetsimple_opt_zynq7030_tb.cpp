// Testbench of justounetsimple_opt_zynq7030: the checks of justounetsimple_opt.
#include "justounetsimple_opt_zynq7030.hpp"

static void run_kernel(cube_word_t *cube, score_word_t *scores, int height, int width) {
    justounetsimple_opt_zynq7030(cube, cube, cube, cube, scores, height, width);
}

#include "../justounetsimple_opt/justounetsimple_opt_check.hpp"
