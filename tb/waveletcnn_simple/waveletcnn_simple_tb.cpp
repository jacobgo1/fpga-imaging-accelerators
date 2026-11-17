// Testbench of waveletcnn_simple: the same checks as waveletcnn_opt (waveletcnn_opt_check.hpp),
// since it must give the same answers bit for bit.
#include "waveletcnn_simple.hpp"
#include <cstdio>

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width) {
    waveletcnn_simple(din, dout, height, width);
}

#include "../waveletcnn_opt/waveletcnn_opt_check.hpp"
