#include "sp_unet_medium_opt_direct.hpp"

#include "../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace weights = sp_unet_medium_int8;

// Every process takes at most T = 34 944 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 8;      // convsp1: 8 bands a cycle, 14 cycles a pixel (14 336), 96 DSPs
constexpr bool ZSCORE_ON_DSPS = false;          // the z-score's 32 multipliers in LUTs: the DSPs are for the convs
constexpr bool CONVSP2_3_ON_DSPS = true;        // convsp2 and convsp3 on DSPs: in LUTs they took ~25 k LUTs (small)
constexpr int CONVSP2_OUT_PER_CYCLE = 1;        // convsp2 24 -> 16: 16 cycles a pixel (16 384), 24 DSPs
constexpr int CONVSP3_OUT_PER_CYCLE = 1;        // convsp3 16 -> 12: 12 (12 288), 16 DSPs
constexpr int FINAL_OUT_PER_CYCLE = 1;          // final 12 -> 3: 3 (3 072), 12 in LUTs
// The convs: channels, cycles a pixel x pixels (+ a row) = cycles a patch, multipliers.
typedef layer_size< 3, 3, true> DOUBLE1_CONV1;  // 12 -> 24   4 x 8 = 32     x 1 057 = 33 824      81
typedef layer_size< 6, 3, true> DOUBLE1_CONV2;  // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef layer_size< 3, 3, true> DOUBLE2_CONV1;  // 24 -> 48   8 x 16 = 128   x 273 = 34 944        81
typedef layer_size< 6, 3, true> DOUBLE2_CONV2;  // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef layer_size< 6, 3, true> UP1_CONV;       // 48 -> 48   8 x 16 = 128   x 273 = 34 944       162
typedef layer_size< 3, 3, true> DOUBLE3_CONV1;  // 48 -> 24   16 x 8 = 128   x 273 = 34 944        81
typedef layer_size< 6, 1, true> DOUBLE3_CONV2;  // 24 -> 24   4 x 24 = 96    x 273 = 26 208        54
typedef layer_size< 6, 3, true> UP2_CONV;       // 24 -> 24   4 x 8 = 32     x 1 057 = 33 824     162
typedef layer_size< 6, 3, true> DOUBLE4_CONV1;  // 48 -> 12   8 x 4 = 32     x 1 057 = 33 824     162
typedef layer_size< 6, 1, true> DOUBLE4_CONV2;  // 12 -> 12   2 x 12 = 24    x 1 057 = 25 368      54
}

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

void sp_unet_medium_opt_direct(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                               hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
                               hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles)
{
    // depth is for co-simulation only.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem4 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
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
