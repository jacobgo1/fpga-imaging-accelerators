// waveletcnn_opt_zynq7030_tb.cpp  --  testbench for the Zynq-7030 version of
// the optimized WaveletCNN kernel: the checks of waveletcnn_opt, the same answers.
#include "waveletcnn_opt_zynq7030.hpp"

static void run_kernel(cube_word_t *din, score_word_t *dout, int height, int width) {
    waveletcnn_opt_zynq7030(din, din, din, din, dout, height, width);
}

#include "../waveletcnn_opt/waveletcnn_opt_check.hpp"
