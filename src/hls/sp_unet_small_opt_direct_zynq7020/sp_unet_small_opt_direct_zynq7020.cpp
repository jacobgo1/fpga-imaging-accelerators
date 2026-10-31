#include "sp_unet_small_opt_direct_zynq7020.hpp"

#include "../../../weights/quantized/sp_unet_small/sp_unet_small_int8.hpp"
namespace weights = sp_unet_small_int8;

// Every process takes at most T = 76 176 cycles a patch; the multipliers of each:
namespace sizes {
constexpr int CONVSP1_BANDS_PER_CYCLE = 20;                         // convsp1: 20 bands a chunk, 6 chunks
constexpr int CONVSP1_PASSES = 12;                                  // ... x 12 passes over its pairs: 73 728 cycles a patch, 20 DSPs
constexpr int SKIP_DEPTH = 2048;                                    // the skip FIFO, pixels (room to spare)
constexpr bool ZSCORE_ON_DSPS = true;                               // the z-score's 4 multipliers on DSPs
constexpr int CONVSP2_IN_PER_CYCLE = 4, CONVSP2_OUT_PER_CYCLE = 1;  // convsp2 24 -> 12: 6 x 12 = 72 cycles a pixel (73 728), 4 DSPs
constexpr bool CONVSP2_ON_DSPS = true;
constexpr int CONVSP3_IN_PER_CYCLE = 2, CONVSP3_OUT_PER_CYCLE = 1;  // convsp3 12 -> 12: 6 x 12 = 72 cycles a pixel (73 728), 2 DSPs
constexpr bool CONVSP3_ON_DSPS = true;
constexpr int FINAL_IN_PER_CYCLE = 1, FINAL_OUT_PER_CYCLE = 1;      // final 8 -> 3: 8 x 3 = 24 cycles a pixel (24 576), 1 DSP
constexpr bool FINAL_ON_DSPS = true;
// The convs: channels, taps x in x out steps a pixel x pixels = cycles a patch, multipliers.
typedef lines_layer_size< 6, 3, 1, true> DOUBLE1_CONV1;             // 12 -> 12  9 x  2 x  4 =   72 a pixel  x 1058 =  76 176   18
typedef lines_layer_size< 6, 3, 1, true> DOUBLE1_CONV2;             // 12 -> 12  9 x  2 x  4 =   72 a pixel  x 1058 =  76 176   18
typedef lines_layer_size<12, 1, 1, true> DOUBLE2_CONV1;             // 12 -> 24  9 x  1 x 24 =  216 a pixel  x  274 =  59 184   12
typedef lines_layer_size<24, 1, 1, true> DOUBLE2_CONV2;             // 24 -> 24  9 x  1 x 24 =  216 a pixel  x  274 =  59 184   24
typedef lines_layer_size<24, 1, 1, true> UP1_CONV;                  // 24 -> 24  9 x  1 x 24 =  216 a pixel  x  274 =  59 184   24
typedef lines_layer_size<12, 1, 1, true> DOUBLE3_CONV1;             // 24 -> 12  9 x  2 x 12 =  216 a pixel  x  274 =  59 184   12
typedef lines_layer_size< 6, 1, 1, true> DOUBLE3_CONV2;             // 12 -> 12  9 x  2 x 12 =  216 a pixel  x  274 =  59 184    6
typedef lines_layer_size< 6, 3, 1, true> UP2_CONV;                  // 12 -> 12  9 x  2 x  4 =   72 a pixel  x 1058 =  76 176   18
typedef lines_layer_size<24, 1, 1, true> DOUBLE4_CONV1;             // 24 ->  8  9 x  1 x  8 =   72 a pixel  x 1058 =  76 176   24
typedef lines_layer_size< 8, 1, 1, true> DOUBLE4_CONV2;             // 8 ->  8  9 x  1 x  8 =   72 a pixel  x 1058 =  76 176    8
}

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

// A conv's template arguments from its size and fractions. Each conv sends
// ReLU'd activations, so the pooling and upsampling after it only take maxima
// and copy (their SHIFT 0), and no 32-bit sums cross a FIFO.
#define LINES_LAYER(size, fracs, layer) sizes::size::IN_PER_CYCLE, sizes::size::OUT_PER_CYCLE, \
    fracs::BIAS_SHIFT, weights::layer##_weight, weights::layer##_bias, sizes::size::DSP, \
    sizes::size::TAPS_PER_STEP, fracs::SHIFT

constexpr int PIXELS_PER_BURST = 8, BURSTS_IN_FLIGHT = 2;   // 240 words a burst

void sp_unet_small_opt_direct_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                                       int height, int width, uint32_t& cycles)
{
    // depth is for co-simulation only.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW
    constexpr int HALF_PATCH = PATCH_SIZE / 2;
    constexpr int PIXELS = PATCH_SIZE * PATCH_SIZE;
    (void)sizes::SKIP_DEPTH;   // only the pragma uses it
    hls::stream<cube_word_t> cube_words[READ_PORTS];
    hls::stream<vec_t<act8_t, sizes::CONVSP1_BANDS_PER_CYCLE>> zscores("zscores");
    hls::stream<vec_t<act_t, STEM1>> convsp1_out("convsp1_out");
    hls::stream<vec_t<act_t, STEM2>> convsp2_out("convsp2_out");
    hls::stream<vec_t<act_t, STEM3>> stem_out("stem_out");
    hls::stream<vec_t<act_t, DOUBLE1>> double1_relu1("double1_relu1"), double1_conv2_out("double1_conv2_out");
    hls::stream<vec_t<act_t, DOUBLE1>> skip("skip"), pool1_out("pool1_out");
    hls::stream<vec_t<act_t, DOUBLE2>> double2_relu1("double2_relu1"), double2_conv2_out("double2_conv2_out");
    hls::stream<vec_t<act_t, DOUBLE2>> pool2_out("pool2_out"), up1_relu("up1_relu");
    hls::stream<vec_t<act_t, DOUBLE3>> double3_relu1("double3_relu1"), double3_conv2_out("double3_conv2_out");
    hls::stream<vec_t<act_t, DOUBLE3>> double3_out("double3_out"), up2_relu("up2_relu");
    hls::stream<vec_t<act_t, DOUBLE3 + DOUBLE1>> joined("joined");
    hls::stream<vec_t<act_t, DOUBLE4>> double4_relu1("double4_relu1"), double4_out("double4_out");
    hls::stream<vec_t<acc_t, CLASSES>> scores("scores");
    hls::stream<bool> all_written("all_written");

    #pragma HLS STREAM variable=cube_words depth=64
    #pragma HLS STREAM variable=zscores    depth=64
    // The skip waits up to a patch for up2; too shallow and it deadlocks.
    #pragma HLS STREAM variable=skip depth=sizes::SKIP_DEPTH
    #pragma HLS BIND_STORAGE variable=skip type=fifo impl=bram
    // The pooled pixels and the upsampled rows come in bursts: room for a burst.
    #pragma HLS STREAM variable=pool2_out   depth=32
    #pragma HLS STREAM variable=double3_out depth=64
    #pragma HLS STREAM variable=scores      depth=32
    #pragma HLS STREAM variable=pool1_out   depth=16
    #pragma HLS BIND_STORAGE variable=pool1_out   type=fifo impl=lutram
    #pragma HLS BIND_STORAGE variable=pool2_out   type=fifo impl=lutram
    #pragma HLS BIND_STORAGE variable=double3_out type=fifo impl=lutram

    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0, PIXELS_PER_BURST, BURSTS_IN_FLIGHT>(
        din0, cube_words[0], height, width);
    pixel_buffer<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD,
                 sizes::CONVSP1_BANDS_PER_CYCLE, READ_PORTS, CONVSP1_CHUNKS, zscore::FRAC, zscore::IN_MAX,
                 zscore::A, zscore::B, sizes::ZSCORE_ON_DSPS>(cube_words, zscores, height, width);

    // The spectral stem, 1x1 convs; the later layers a few inputs a step, on a few DSPs.
    pointwise_wide<PATCH_SIZE, PATCH_SIZE, PIXELS, MODEL_BANDS, STEM1, sizes::CONVSP1_BANDS_PER_CYCLE,
                   CONVSP1_FRACS::BIAS_SHIFT, CONVSP1_FRACS::SHIFT, weights::convsp1_weight, weights::convsp1_bias,
                   sizes::CONVSP1_PASSES>(zscores, convsp1_out, height, width);
    pointwise<PATCH_SIZE, PATCH_SIZE, PIXELS, STEM1, STEM2, sizes::CONVSP2_OUT_PER_CYCLE, 1, true,
              CONVSP2_FRACS::BIAS_SHIFT, CONVSP2_FRACS::SHIFT, weights::convsp2_weight, weights::convsp2_bias,
              sizes::CONVSP2_ON_DSPS, sizes::CONVSP2_IN_PER_CYCLE>(convsp1_out, convsp2_out, height, width);
    pointwise<PATCH_SIZE, PATCH_SIZE, PIXELS, STEM2, STEM3, sizes::CONVSP3_OUT_PER_CYCLE, 1, true,
              CONVSP3_FRACS::BIAS_SHIFT, CONVSP3_FRACS::SHIFT, weights::convsp3_weight, weights::convsp3_bias,
              sizes::CONVSP3_ON_DSPS, sizes::CONVSP3_IN_PER_CYCLE>(convsp2_out, stem_out, height, width);

    // The encoder: double1 at 32 x 32, whose output is also the skip; double2 at 16 x 16.
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, STEM3, DOUBLE1,
                  LINES_LAYER(DOUBLE1_CONV1, DOUBLE1_CONV1_FRACS, double1_block_0)>(
        stem_out, double1_relu1, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE1, DOUBLE1,
                  LINES_LAYER(DOUBLE1_CONV2, DOUBLE1_CONV2_FRACS, double1_block_3)>(
        double1_relu1, double1_conv2_out, height, width);
    relu_pool_skip<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 0>(double1_conv2_out, skip, pool1_out, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE1, DOUBLE2,
                  LINES_LAYER(DOUBLE2_CONV1, DOUBLE2_CONV1_FRACS, double2_block_0)>(
        pool1_out, double2_relu1, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                  LINES_LAYER(DOUBLE2_CONV2, DOUBLE2_CONV2_FRACS, double2_block_3)>(
        double2_relu1, double2_conv2_out, height, width);
    // Pool to 8 x 8 and upsample back: each 2x2 block becomes its maximum.
    relu_pool_up<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, 0>(double2_conv2_out, pool2_out, height, width);

    // The decoder: up1's conv at 16 x 16, double3, upsample to 32 x 32, up2's conv.
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                  LINES_LAYER(UP1_CONV, UP1_FRACS, up1_1)>(pool2_out, up1_relu, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE3,
                  LINES_LAYER(DOUBLE3_CONV1, DOUBLE3_CONV1_FRACS, double3_block_0)>(
        up1_relu, double3_relu1, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE3, DOUBLE3,
                  LINES_LAYER(DOUBLE3_CONV2, DOUBLE3_CONV2_FRACS, double3_block_3)>(
        double3_relu1, double3_conv2_out, height, width);
    relu_up<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, 0>(double3_conv2_out, double3_out, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3, DOUBLE3,
                  LINES_LAYER(UP2_CONV, UP2_FRACS, up2_1)>(double3_out, up2_relu, height, width);

    // The skip joins, then double4 and the final 1x1 conv: the scores.
    concat_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE3, DOUBLE1>(up2_relu, skip, joined, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3 + DOUBLE1, DOUBLE4,
                  LINES_LAYER(DOUBLE4_CONV1, DOUBLE4_CONV1_FRACS, double4_block_0)>(
        joined, double4_relu1, height, width);
    conv3x3_lines<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE4, DOUBLE4,
                  LINES_LAYER(DOUBLE4_CONV2, DOUBLE4_CONV2_FRACS, double4_block_3)>(
        double4_relu1, double4_out, height, width);
    pointwise<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE4, CLASSES, sizes::FINAL_OUT_PER_CYCLE, 1, false,
              FINAL_FRACS::BIAS_SHIFT, FINAL_FRACS::SHIFT, weights::final_weight, weights::final_bias,
              sizes::FINAL_ON_DSPS, sizes::FINAL_IN_PER_CYCLE>(double4_out, scores, height, width);
    store_scores<PATCH_SIZE, PATCH_SIZE, FINAL_FRACS::SHIFT>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
