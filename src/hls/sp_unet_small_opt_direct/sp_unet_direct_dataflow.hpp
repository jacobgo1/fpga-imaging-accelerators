#pragma once

// The direct chain, shared by sp_unet_small_opt_direct, _medium_ and _large_:
// sp_unet_small_opt without Winograd, every 3x3 conv done directly
// (conv3x3_grouped), the maps in raster order, the pooling and upsampling in
// justounetsimple_opt's ReLU modules. The same answers; simpler, slower for the
// same DSPs. The kernel's .cpp first includes its model's weights as namespace
// weights and its size table as namespace sizes.

#include "../sp_unet_small_opt/sp_unet_opt_common.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/pixel_buffer.hpp"
#include "../../optimized/zcu104/pointwise_wide.hpp"
#include "../../optimized/shared/pointwise.hpp"
#include "../../optimized/shared/conv3x3_grouped.hpp"
#include "../../optimized/shared/relu_act.hpp"
#include "../../optimized/shared/relu_pool_skip.hpp"
#include "../../optimized/shared/relu_pool_up.hpp"
#include "../../optimized/shared/relu_up.hpp"
#include "../../optimized/shared/concat_pixels.hpp"
#include "../../optimized/shared/store_scores.hpp"
#include "../../optimized/shared/count_cycles.hpp"

// A conv's template arguments from its size and fractions.
#define CONV_LAYER(size, fracs, layer) sizes::size::IN_PER_CYCLE, sizes::size::OUT_PER_CYCLE, \
    fracs::BIAS_SHIFT, weights::layer##_weight, weights::layer##_bias, sizes::size::DSP

static void sp_unet_direct_dataflow(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                                    hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                                    hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles)
{
    #pragma HLS DATAFLOW
    constexpr int HALF_PATCH = PATCH_SIZE / 2;
    constexpr int PIXELS = PATCH_SIZE * PATCH_SIZE, HALF_PIXELS = HALF_PATCH * HALF_PATCH;
    hls::stream<cube_word_t> cube_words[READ_PORTS];
    hls::stream<vec_t<act8_t, sizes::CONVSP1_BANDS_PER_CYCLE>> zscores("zscores");
    hls::stream<vec_t<act_t, STEM1>> convsp1_out("convsp1_out");
    hls::stream<vec_t<act_t, STEM2>> convsp2_out("convsp2_out");
    hls::stream<vec_t<act_t, STEM3>> stem_out("stem_out");
    hls::stream<vec_t<acc_t, DOUBLE1>> double1_conv1_out("double1_conv1_out"), double1_conv2_out("double1_conv2_out");
    hls::stream<vec_t<act_t, DOUBLE1>> double1_relu1("double1_relu1"), skip("skip"), pool1_out("pool1_out");
    hls::stream<vec_t<acc_t, DOUBLE2>> double2_conv1_out("double2_conv1_out"), double2_conv2_out("double2_conv2_out");
    hls::stream<vec_t<acc_t, DOUBLE2>> up1_conv_out("up1_conv_out");
    hls::stream<vec_t<act_t, DOUBLE2>> double2_relu1("double2_relu1"), pool2_out("pool2_out"), up1_relu("up1_relu");
    hls::stream<vec_t<acc_t, DOUBLE3>> double3_conv1_out("double3_conv1_out"), double3_conv2_out("double3_conv2_out");
    hls::stream<vec_t<acc_t, DOUBLE3>> up2_conv_out("up2_conv_out");
    hls::stream<vec_t<act_t, DOUBLE3>> double3_relu1("double3_relu1"), double3_out("double3_out"), up2_relu("up2_relu");
    hls::stream<vec_t<act_t, DOUBLE3 + DOUBLE1>> joined("joined");
    hls::stream<vec_t<acc_t, DOUBLE4>> double4_conv1_out("double4_conv1_out"), double4_conv2_out("double4_conv2_out");
    hls::stream<vec_t<act_t, DOUBLE4>> double4_relu1("double4_relu1"), double4_out("double4_out");
    hls::stream<vec_t<acc_t, CLASSES>> scores("scores");
    hls::stream<bool> all_written("all_written");

    // pixel_buffer takes a strip's words in 480 of each 896 steps: room for a
    // strip row lets the readers use the other 416 too (11 503 to 9 856 cycles a patch).
    #pragma HLS STREAM variable=cube_words depth=512
    #pragma HLS BIND_STORAGE variable=cube_words type=fifo impl=bram
    #pragma HLS STREAM variable=zscores depth=64
    // The skip waits up to a patch for up2; too shallow and it deadlocks.
    #pragma HLS STREAM variable=skip depth=2048
    #pragma HLS BIND_STORAGE variable=skip type=fifo impl=uram
    // The pooled pixels and the upsampled rows come in bursts: room for a burst.
    #pragma HLS STREAM variable=pool1_out   depth=16
    #pragma HLS STREAM variable=pool2_out   depth=32
    #pragma HLS STREAM variable=double3_out depth=64
    #pragma HLS BIND_STORAGE variable=pool1_out   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=pool2_out   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double3_out type=fifo impl=srl
    #pragma HLS STREAM variable=scores depth=32
    // 16 deep between the layers, not 2: they run at the same rate, and a pipeline
    // freezes all its stages when one read or write waits, so one conv's short
    // wait froze the next (11 503 cycles a patch with 2, 9 856 with 16).
    #pragma HLS STREAM variable=convsp1_out       depth=16
    #pragma HLS STREAM variable=convsp2_out       depth=16
    #pragma HLS STREAM variable=stem_out          depth=16
    #pragma HLS STREAM variable=double1_conv1_out depth=16
    #pragma HLS STREAM variable=double1_relu1     depth=16
    #pragma HLS STREAM variable=double1_conv2_out depth=16
    #pragma HLS STREAM variable=double2_conv1_out depth=16
    #pragma HLS STREAM variable=double2_relu1     depth=16
    #pragma HLS STREAM variable=double2_conv2_out depth=16
    #pragma HLS STREAM variable=up1_conv_out      depth=16
    #pragma HLS STREAM variable=up1_relu          depth=16
    #pragma HLS STREAM variable=double3_conv1_out depth=16
    #pragma HLS STREAM variable=double3_relu1     depth=16
    #pragma HLS STREAM variable=double3_conv2_out depth=16
    #pragma HLS STREAM variable=up2_conv_out      depth=16
    #pragma HLS STREAM variable=up2_relu          depth=16
    #pragma HLS STREAM variable=joined            depth=16
    #pragma HLS STREAM variable=double4_conv1_out depth=16
    #pragma HLS STREAM variable=double4_relu1     depth=16
    #pragma HLS STREAM variable=double4_conv2_out depth=16
    #pragma HLS STREAM variable=double4_out       depth=16
    #pragma HLS BIND_STORAGE variable=convsp1_out       type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=convsp2_out       type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stem_out          type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double1_conv1_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double1_relu1     type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double1_conv2_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double2_conv1_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double2_relu1     type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double2_conv2_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=up1_conv_out      type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=up1_relu          type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double3_conv1_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double3_relu1     type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double3_conv2_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=up2_conv_out      type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=up2_relu          type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=joined            type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double4_conv1_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double4_relu1     type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double4_conv2_out type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double4_out       type=fifo impl=srl

    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0>(din0, cube_words[0], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 1>(din1, cube_words[1], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 2>(din2, cube_words[2], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 3>(din3, cube_words[3], height, width);
    pixel_buffer<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD,
                 sizes::CONVSP1_BANDS_PER_CYCLE, READ_PORTS, CONVSP1_CHUNKS, zscore::FRAC, zscore::IN_MAX,
                 zscore::A, zscore::B, sizes::ZSCORE_ON_DSPS>(cube_words, zscores, height, width);

    // The spectral stem, 1x1 convs; convsp1 two multiplies per DSP.
    pointwise_wide<PATCH_SIZE, PATCH_SIZE, PIXELS, MODEL_BANDS, STEM1, sizes::CONVSP1_BANDS_PER_CYCLE,
                   CONVSP1_FRACS::BIAS_SHIFT, CONVSP1_FRACS::SHIFT, weights::convsp1_weight, weights::convsp1_bias>(
        zscores, convsp1_out, height, width);
    pointwise<PATCH_SIZE, PATCH_SIZE, PIXELS, STEM1, STEM2, sizes::CONVSP2_OUT_PER_CYCLE, 1, true,
              CONVSP2_FRACS::BIAS_SHIFT, CONVSP2_FRACS::SHIFT, weights::convsp2_weight, weights::convsp2_bias,
              sizes::CONVSP2_3_ON_DSPS>(convsp1_out, convsp2_out, height, width);
    pointwise<PATCH_SIZE, PATCH_SIZE, PIXELS, STEM2, STEM3, sizes::CONVSP3_OUT_PER_CYCLE, 1, true,
              CONVSP3_FRACS::BIAS_SHIFT, CONVSP3_FRACS::SHIFT, weights::convsp3_weight, weights::convsp3_bias,
              sizes::CONVSP2_3_ON_DSPS>(convsp2_out, stem_out, height, width);

    // The encoder: double1 at 32 x 32, whose output is also the skip; double2 at 16 x 16.
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, STEM3, DOUBLE1,
                    CONV_LAYER(DOUBLE1_CONV1, DOUBLE1_CONV1_FRACS, double1_block_0)>(
        stem_out, double1_conv1_out, height, width);
    relu_act<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE1_CONV1_FRACS::SHIFT, DOUBLE1>(
        double1_conv1_out, double1_relu1, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE1, DOUBLE1,
                    CONV_LAYER(DOUBLE1_CONV2, DOUBLE1_CONV2_FRACS, double1_block_3)>(
        double1_relu1, double1_conv2_out, height, width);
    relu_pool_skip<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE1_CONV2_FRACS::SHIFT>(
        double1_conv2_out, skip, pool1_out, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE1, DOUBLE2,
                    CONV_LAYER(DOUBLE2_CONV1, DOUBLE2_CONV1_FRACS, double2_block_0)>(
        pool1_out, double2_conv1_out, height, width);
    relu_act<PATCH_SIZE, PATCH_SIZE, HALF_PIXELS, DOUBLE2_CONV1_FRACS::SHIFT, DOUBLE2>(
        double2_conv1_out, double2_relu1, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                    CONV_LAYER(DOUBLE2_CONV2, DOUBLE2_CONV2_FRACS, double2_block_3)>(
        double2_relu1, double2_conv2_out, height, width);
    // Pool to 8 x 8 and upsample back: each 2x2 block becomes its maximum.
    relu_pool_up<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2_CONV2_FRACS::SHIFT>(
        double2_conv2_out, pool2_out, height, width);

    // The decoder: up1's conv at 16 x 16, double3, upsample to 32 x 32, up2's conv.
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                    CONV_LAYER(UP1_CONV, UP1_FRACS, up1_1)>(pool2_out, up1_conv_out, height, width);
    relu_act<PATCH_SIZE, PATCH_SIZE, HALF_PIXELS, UP1_FRACS::SHIFT, DOUBLE2>(up1_conv_out, up1_relu, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE3,
                    CONV_LAYER(DOUBLE3_CONV1, DOUBLE3_CONV1_FRACS, double3_block_0)>(
        up1_relu, double3_conv1_out, height, width);
    relu_act<PATCH_SIZE, PATCH_SIZE, HALF_PIXELS, DOUBLE3_CONV1_FRACS::SHIFT, DOUBLE3>(
        double3_conv1_out, double3_relu1, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE3, DOUBLE3,
                    CONV_LAYER(DOUBLE3_CONV2, DOUBLE3_CONV2_FRACS, double3_block_3)>(
        double3_relu1, double3_conv2_out, height, width);
    relu_up<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE3_CONV2_FRACS::SHIFT>(
        double3_conv2_out, double3_out, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3, DOUBLE3,
                    CONV_LAYER(UP2_CONV, UP2_FRACS, up2_1)>(double3_out, up2_conv_out, height, width);
    relu_act<PATCH_SIZE, PATCH_SIZE, PIXELS, UP2_FRACS::SHIFT, DOUBLE3>(up2_conv_out, up2_relu, height, width);

    // The skip joins, then double4 and the final 1x1 conv: the scores.
    concat_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE3, DOUBLE1>(up2_relu, skip, joined, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3 + DOUBLE1, DOUBLE4,
                    CONV_LAYER(DOUBLE4_CONV1, DOUBLE4_CONV1_FRACS, double4_block_0)>(
        joined, double4_conv1_out, height, width);
    relu_act<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE4_CONV1_FRACS::SHIFT, DOUBLE4>(
        double4_conv1_out, double4_relu1, height, width);
    conv3x3_grouped<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE4, DOUBLE4,
                    CONV_LAYER(DOUBLE4_CONV2, DOUBLE4_CONV2_FRACS, double4_block_3)>(
        double4_relu1, double4_conv2_out, height, width);
    relu_act<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE4_CONV2_FRACS::SHIFT, DOUBLE4>(
        double4_conv2_out, double4_out, height, width);
    pointwise<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE4, CLASSES, sizes::FINAL_OUT_PER_CYCLE, 1, false,
              FINAL_FRACS::BIAS_SHIFT, FINAL_FRACS::SHIFT, weights::final_weight, weights::final_bias, false>(
        double4_out, scores, height, width);
    store_scores<PATCH_SIZE, PATCH_SIZE, FINAL_FRACS::SHIFT>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}

#undef CONV_LAYER
