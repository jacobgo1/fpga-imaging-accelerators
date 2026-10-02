#include "justounet2d_golden.hpp"

void justounet2d_golden_top(
    data_t din[JUNET2D_H][JUNET2D_W][JUNET2D_IN_CH],
    data_t w1[JUNET2D_BASE_CH][JUNET2D_IN_CH][JUNET2D_K][JUNET2D_K],
    data_t b1[JUNET2D_BASE_CH],
    data_t bn1_weight[JUNET2D_BASE_CH], data_t bn1_bias[JUNET2D_BASE_CH],
    data_t bn1_mean[JUNET2D_BASE_CH],   data_t bn1_var[JUNET2D_BASE_CH],
    data_t w2[2*JUNET2D_BASE_CH][JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K],
    data_t b2[2*JUNET2D_BASE_CH],
    data_t bn2_weight[2*JUNET2D_BASE_CH], data_t bn2_bias[2*JUNET2D_BASE_CH],
    data_t bn2_mean[2*JUNET2D_BASE_CH],   data_t bn2_var[2*JUNET2D_BASE_CH],
    data_t w3[JUNET2D_BASE_CH][2*JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K],
    data_t b3[JUNET2D_BASE_CH],
    data_t bn3_weight[JUNET2D_BASE_CH], data_t bn3_bias[JUNET2D_BASE_CH],
    data_t bn3_mean[JUNET2D_BASE_CH],   data_t bn3_var[JUNET2D_BASE_CH],
    data_t w4[JUNET2D_OUT_CH][JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K],
    data_t b4[JUNET2D_OUT_CH],
    data_t bn4_weight[JUNET2D_OUT_CH], data_t bn4_bias[JUNET2D_OUT_CH],
    data_t bn4_mean[JUNET2D_OUT_CH],   data_t bn4_var[JUNET2D_OUT_CH],
    data_t eps,
    data_t dout[JUNET2D_H][JUNET2D_W][JUNET2D_OUT_CH])
{
    // One shared AXI master (bundle=gmem) rather than one per array, plus
    // an s_axilite entry per array: the same pattern justoliunet.cpp
    // uses, and for the same reason -- unlike the 14 base-module
    // *_golden kernels (which use mode=bram, fine for an isolated
    // synthesis/cosim baseline with no real weights to load), this
    // kernel IS the deployment target. Real trained weights have to be
    // DMA'd out of wherever the PS put them in DDR, at addresses chosen
    // at run time, not fixed on-chip BRAMs -- see software/zcu104/
    // export_justounet2d_weights.py + run_justounet2d.py for the PS side
    // that writes those buffers and points these registers at them.
    #pragma HLS INTERFACE mode=m_axi port=din         bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w1          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b1          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn1_weight  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn1_bias    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn1_mean    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn1_var     bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w2          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b2          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn2_weight  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn2_bias    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn2_mean    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn2_var     bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w3          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b3          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn3_weight  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn3_bias    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn3_mean    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn3_var     bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=w4          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=b4          bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn4_weight  bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn4_bias    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn4_mean    bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=bn4_var     bundle=gmem offset=slave
    #pragma HLS INTERFACE mode=m_axi port=dout        bundle=gmem offset=slave

    #pragma HLS INTERFACE mode=s_axilite port=din         bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w1          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b1          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn1_weight  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn1_bias    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn1_mean    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn1_var     bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w2          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b2          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn2_weight  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn2_bias    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn2_mean    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn2_var     bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w3          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b3          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn3_weight  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn3_bias    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn3_mean    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn3_var     bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=w4          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=b4          bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn4_weight  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn4_bias    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn4_mean    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=bn4_var     bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout        bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=eps         bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return      bundle=control

    // Deliberately just this: the same call the testbench's expected
    // reference makes (see this file's header comment). No staging
    // buffers, no partitioning, no directives -- this is the baseline.
    justounet2d_golden<data_t, JUNET2D_H, JUNET2D_W, JUNET2D_IN_CH,
                        JUNET2D_BASE_CH, JUNET2D_OUT_CH, JUNET2D_K>(
        din,
        w1, b1, bn1_weight, bn1_bias, bn1_mean, bn1_var,
        w2, b2, bn2_weight, bn2_bias, bn2_mean, bn2_var,
        w3, b3, bn3_weight, bn3_bias, bn3_mean, bn3_var,
        w4, b4, bn4_weight, bn4_bias, bn4_mean, bn4_var,
        eps,
        dout);
}
