#pragma once

// The direct SpecPW-UNet's DATAFLOW chain for the Zynq-7020, shared by
// sp_unet_small_opt_direct_zynq7020, _medium_ and _large_. Before including it,
// a kernel's .cpp includes sp_unet_opt_zynq7020.hpp (the design), its model's
// int8 weights as namespace q and its table of sizes as namespace cfg.

#include "../sp_unet_small_opt/sp_unet_opt_common.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/pixel_buffer.hpp"
#include "../../optimized/zynq7020/pointwise_wide.hpp"
#include "../../optimized/shared/pointwise.hpp"
#include "../../optimized/shared/conv3x3_lines.hpp"
#include "../../optimized/shared/relu_pool_skip.hpp"
#include "../../optimized/shared/relu_pool_up.hpp"
#include "../../optimized/shared/relu_up.hpp"
#include "../../optimized/shared/concat_pixels.hpp"
#include "../../optimized/shared/store_scores.hpp"
#include "../../optimized/shared/count_cycles.hpp"

// A conv layer's template arguments from its table entry and fractions. Every
// conv keeps its 3x3 window in three row memories (conv3x3_lines: the window
// registers alone would not fit the 7020's flip-flops on medium and large) and sends ReLU'd activations, rounded to ACT_FRAC as each output group
// finishes: the pooling and upsampling after double1b, double2b and double3b
// only take maxima and copy (their SHIFT 0), and no 32-bit sums cross a FIFO.
#define SPU7_CONV(L, F, layer) cfg::L::PI, cfg::L::PO, F::BIAS_SHIFT, q::layer##_weight, \
                               q::layer##_bias, cfg::L::DSP, cfg::L::KT, F::SHIFT

static void sp_unet_direct_zynq7020_dataflow(hls::burst_maxi<spu_raw_t> din0, hls::burst_maxi<spu_out_t> dout,
                                             int height, int width, uint32_t& cycles)
{
#pragma HLS DATAFLOW
    constexpr int H2 = H / 2, W2 = W / 2;
    hls::stream<spu_raw_t> s_raw[PORTS];
    hls::stream<vec_t<act8_t, cfg::P1> > s_bands("s_bands");
    hls::stream<vec_t<act_t, S1> > s_sp1("s_sp1");
    hls::stream<vec_t<act_t, S2> > s_sp2("s_sp2");
    hls::stream<vec_t<act_t, S3> > s_stem("s_stem");
    hls::stream<vec_t<act_t, C1> > s_a1("s_a1"), s_d1b("s_d1b"), s_skip("s_skip"), s_pool1("s_pool1");
    hls::stream<vec_t<act_t, C2> > s_a2("s_a2"), s_d2b("s_d2b"), s_pool2("s_pool2"), s_u1("s_u1");
    hls::stream<vec_t<act_t, C3> > s_a3("s_a3"), s_d3b("s_d3b"), s_dec3("s_dec3"), s_u2("s_u2");
    hls::stream<vec_t<act_t, C3 + C1> > s_cat("s_cat");
    hls::stream<vec_t<act_t, C4> > s_a4("s_a4"), s_dec4("s_dec4");
    hls::stream<vec_t<acc_t, CLASSES> > s_scores("s_scores");
    hls::stream<bool> s_finished("s_finished");
// The reader pauses between bursts; let it run ahead.
#pragma HLS STREAM variable = s_raw depth = 64
#pragma HLS STREAM variable = s_bands depth = 64
// The skip connection: double1's activations wait here until up2's come back,
// less than a patch later (1 024 pixels), each patch being done on its own; too
// shallow and the encoder stops while the decoder waits for it, a deadlock.
// No UltraRAM on the 7020: block RAM.
#pragma HLS STREAM variable = s_skip depth = 2048
#pragma HLS BIND_STORAGE variable = s_skip type = fifo impl = bram
// relu_pool_up gives a row pair (32 pixels) at once, relu_up two rows (64).
#pragma HLS STREAM variable = s_pool2 depth = 32
#pragma HLS STREAM variable = s_dec3 depth = 64
// A patch row of scores, so the final layer keeps going while store starts a burst.
#pragma HLS STREAM variable = s_scores depth = 32
// Between the layers, 16 deep, not 2: the convs run at the same rate, and an
// HLS pipeline freezes all its stages when one read or write must wait, so
// with 2 a conv's short wait froze the next (7 % in the cycle model of small;
// 17 % on the ZCU104's board). In LUTRAM, cheaper per bit than shift registers.
#pragma HLS STREAM variable = s_sp1 depth = 16
#pragma HLS STREAM variable = s_sp2 depth = 16
#pragma HLS STREAM variable = s_stem depth = 16
#pragma HLS STREAM variable = s_a1 depth = 16
#pragma HLS STREAM variable = s_d1b depth = 16
#pragma HLS STREAM variable = s_pool1 depth = 16
#pragma HLS STREAM variable = s_a2 depth = 16
#pragma HLS STREAM variable = s_d2b depth = 16
#pragma HLS STREAM variable = s_u1 depth = 16
#pragma HLS STREAM variable = s_a3 depth = 16
#pragma HLS STREAM variable = s_d3b depth = 16
#pragma HLS STREAM variable = s_u2 depth = 16
#pragma HLS STREAM variable = s_cat depth = 16
#pragma HLS STREAM variable = s_a4 depth = 16
#pragma HLS STREAM variable = s_dec4 depth = 16
#pragma HLS BIND_STORAGE variable = s_sp1 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_sp2 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_stem type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_a1 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_d1b type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_pool1 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_a2 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_d2b type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_pool2 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_u1 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_a3 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_d3b type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_dec3 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_u2 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_cat type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_a4 type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = s_dec4 type = fifo impl = lutram

    // A patch row is 32 x 30 = 960 words: four bursts of 8 pixels (240 <= 256),
    // the next requested while the current one streams in.
    read_rows_ahead<H, W, WORDS, PORTS, 0, 8, 2>(din0, s_raw[0], height, width);
    // The z-score, then each pixel's P1-band chunks, raster order.
    pixel_buffer<H, W, WORDS, FIRST, PORT_BANDS, cfg::P1, PORTS, CHUNKS, prep::FRAC, prep::IN_MAX,
                 prep::A, prep::B, cfg::ZSCORE_DSP>(s_raw, s_bands, height, width);

    // The spectral stem, 1x1 convs per pixel; convsp2/3 and the final layer PI inputs a step,
    // on a few DSPs: they have time to spare.
    pointwise_wide<H, W, H * W, IN_CH, S1, cfg::P1, F_SP1::BIAS_SHIFT, F_SP1::SHIFT,
                   q::convsp1_weight, q::convsp1_bias, cfg::SP1_PASSES>(s_bands, s_sp1, height, width);
    pointwise<H, W, H * W, S1, S2, cfg::SP2_PO, 1, true, F_SP2::BIAS_SHIFT, F_SP2::SHIFT,
              q::convsp2_weight, q::convsp2_bias, cfg::SP2_DSP, cfg::SP2_PI>(s_sp1, s_sp2, height, width);
    pointwise<H, W, H * W, S2, S3, cfg::SP3_PO, 1, true, F_SP3::BIAS_SHIFT, F_SP3::SHIFT,
              q::convsp3_weight, q::convsp3_bias, cfg::SP3_DSP, cfg::SP3_PI>(s_sp2, s_stem, height, width);

    // The encoder: double1 at 32 x 32 (its output also the skip), double2 at 16 x 16.
    conv3x3_lines<H, W, H, W, S3, C1, SPU7_CONV(D1A, F_D1A, double1_block_0)>(s_stem, s_a1, height, width);
    conv3x3_lines<H, W, H, W, C1, C1, SPU7_CONV(D1B, F_D1B, double1_block_3)>(s_a1, s_d1b, height, width);
    relu_pool_skip<H, W, H, W, 0>(s_d1b, s_skip, s_pool1, height, width);   // -> 16 x 16
    conv3x3_lines<H, W, H2, W2, C1, C2, SPU7_CONV(D2A, F_D2A, double2_block_0)>(s_pool1, s_a2, height, width);
    conv3x3_lines<H, W, H2, W2, C2, C2, SPU7_CONV(D2B, F_D2B, double2_block_3)>(s_a2, s_d2b, height, width);
    // pool to 8 x 8, then nearest up to 16 x 16: each 2x2 block its maximum
    relu_pool_up<H, W, H2, W2, 0>(s_d2b, s_pool2, height, width);

    // The decoder: up1's conv at 16 x 16, double3, up to 32 x 32, up2's conv.
    conv3x3_lines<H, W, H2, W2, C2, C2, SPU7_CONV(UP1, F_UP1, up1_1)>(s_pool2, s_u1, height, width);
    conv3x3_lines<H, W, H2, W2, C2, C3, SPU7_CONV(D3A, F_D3A, double3_block_0)>(s_u1, s_a3, height, width);
    conv3x3_lines<H, W, H2, W2, C3, C3, SPU7_CONV(D3B, F_D3B, double3_block_3)>(s_a3, s_d3b, height, width);
    relu_up<H, W, H2, W2, 0>(s_d3b, s_dec3, height, width);               // -> 32 x 32
    conv3x3_lines<H, W, H, W, C3, C3, SPU7_CONV(UP2, F_UP2, up2_1)>(s_dec3, s_u2, height, width);

    // The skip connection, then double4 and the final 1x1 conv: the scores.
    concat_pixels<H, W, H * W, C3, C1>(s_u2, s_skip, s_cat, height, width);
    conv3x3_lines<H, W, H, W, C3 + C1, C4, SPU7_CONV(D4A, F_D4A, double4_block_0)>(s_cat, s_a4, height, width);
    conv3x3_lines<H, W, H, W, C4, C4, SPU7_CONV(D4B, F_D4B, double4_block_3)>(s_a4, s_dec4, height, width);
    pointwise<H, W, H * W, C4, CLASSES, cfg::FIN_PO, 1, false, F_FIN::BIAS_SHIFT, F_FIN::SHIFT,
              q::final_weight, q::final_bias, cfg::FIN_DSP, cfg::FIN_PI>(s_dec4, s_scores, height, width);
    store_scores<H, W, F_FIN::SHIFT>(s_scores, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (count_cycles.hpp).
    count_cycles(s_finished, cycles);
}

#undef SPU7_CONV
