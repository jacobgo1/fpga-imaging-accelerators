// Testbench of justounetsimple_simple: the same checks as justounetsimple_opt
// (justounetsimple_opt_check.hpp), since it must give the same answers bit for bit.
#include "justounetsimple_simple.hpp"
#include <cstdio>

static void run_kernel(cube_word_t *cube, score_word_t *scores, int height, int width) {
    justounetsimple_simple(cube, scores, height, width);
}

#include "../justounetsimple_opt/justounetsimple_opt_check.hpp"
