#include "sp_unet_small_opt_u8_zynq7020.hpp"
#include "../../../../optimized/read_patch_rows_simple.hpp"
#include "../../../../optimized/pixel_buffer.hpp"
#include "../../../../optimized/pointwise_packed.hpp"
#include "../../../../optimized/conv1x1.hpp"
#include "../../../../optimized/conv3x3_tiled_ram.hpp"
#include "../../../../optimized/max_pool.hpp"
#include "../../../../optimized/duplicate.hpp"
#include "../../../../optimized/upsample.hpp"
#include "../../../../optimized/concat_pixels.hpp"
#include "../../../../optimized/store_scores.hpp"
#include "../../../../optimized/count_cycles.hpp"

#include "../../../../../weights/quantized/sp_unet_small/sp_unet_small_int8.hpp"
namespace weights = sp_unet_small_int8;

#include "../sp_unet_small_opt_u8/sp_unet_opt_u8_common.hpp"

void sp_unet_small_opt_u8_zynq7020(const cube_word_t* din0, hls::burst_maxi<score_word_t> dout,
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
    hls::stream<cube_word_t> cube_words[READ_PORTS];
    #pragma HLS STREAM variable=cube_words depth=64
    hls::stream<vec_t<act8_t, sizes::CONVSP1_BANDS_PER_CYCLE>, 64> zscores("zscores");
    // Every activation uint8 (0 .. 255, its layer's fraction bits); 2 deep between the layers.
    hls::stream<vec_t<uact_t, STEM1>> convsp1_out("convsp1_out");
    hls::stream<vec_t<uact_t, STEM2>> convsp2_out("convsp2_out");
    hls::stream<vec_t<uact_t, STEM3>> stem_out("stem_out");
    hls::stream<vec_t<uact_t, DOUBLE1>> double1_relu1("double1_relu1"), double1_out("double1_out");
    // The skip waits up to a patch for up2; too shallow and it deadlocks.
    hls::stream<vec_t<uact_t, DOUBLE1>, sizes::SKIP_DEPTH> skip("skip");
    #pragma HLS BIND_STORAGE variable=skip type=fifo impl=bram
    // The pooled pixels and the upsampled rows come in bursts: room for a burst.
    hls::stream<vec_t<uact_t, DOUBLE1>> pool1_in("pool1_in");
    hls::stream<vec_t<uact_t, DOUBLE1>, 16> pool1_out("pool1_out");
    hls::stream<vec_t<uact_t, DOUBLE2>> double2_relu1("double2_relu1"), double2_out("double2_out");
    hls::stream<vec_t<uact_t, DOUBLE2>> pool2_out("pool2_out");
    hls::stream<vec_t<uact_t, DOUBLE2>, 32> pool2_up("pool2_up");
    hls::stream<vec_t<uact_t, DOUBLE2>> up1_relu("up1_relu");
    hls::stream<vec_t<uact_t, DOUBLE3>> double3_relu1("double3_relu1"), double3_act("double3_act");
    hls::stream<vec_t<uact_t, DOUBLE3>, 64> double3_out("double3_out");
    hls::stream<vec_t<uact_t, DOUBLE3>> up2_relu("up2_relu");
    hls::stream<vec_t<uact_t, DOUBLE3 + DOUBLE1>> joined("joined");
    hls::stream<vec_t<uact_t, DOUBLE4>> double4_relu1("double4_relu1"), double4_out("double4_out");
    hls::stream<vec_t<sum_t, CLASSES>, 32> scores("scores");
    hls::stream<bool> all_written("all_written");
    #pragma HLS BIND_STORAGE variable=pool1_out   type=fifo impl=lutram
    #pragma HLS BIND_STORAGE variable=pool2_up    type=fifo impl=lutram
    #pragma HLS BIND_STORAGE variable=double3_out type=fifo impl=lutram

    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0>(din0, cube_words[0], height, width);
    pixel_buffer<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD,
                 sizes::CONVSP1_BANDS_PER_CYCLE, READ_PORTS, CONVSP1_CHUNKS, zscore::FRAC, zscore::IN_MAX,
                 zscore::A, zscore::B, sizes::ZSCORE_ON_DSPS>(cube_words, zscores, height, width);

    // The spectral stem, 1x1 convs, uint8 out; convsp1 two multiplies per DSP (int8 z-scores in),
    // the later layers a few inputs a step.
    pointwise_packed<PATCH_SIZE, PATCH_SIZE, PIXELS, MODEL_BANDS, STEM1, sizes::CONVSP1_BANDS_PER_CYCLE,
                     u8::CONVSP1::BIAS_SHIFT, u8::CONVSP1::SHIFT, weights::convsp1_weight, weights::convsp1_bias,
                     DSP_PACK_SHIFT, sizes::CONVSP1_PASSES, activation::relu8>(zscores, convsp1_out, height, width);
    conv1x1<patch_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS>, STEM1, STEM2, sizes::CONVSP2_OUT_PER_CYCLE, activation::relu8,
            u8::CONVSP2::BIAS_SHIFT, u8::CONVSP2::SHIFT, weights::convsp2_weight, weights::convsp2_bias,
            sizes::CONVSP2_ON_DSPS, sizes::CONVSP2_IN_PER_CYCLE>(convsp1_out, convsp2_out, height, width);
    conv1x1<patch_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS>, STEM2, STEM3, sizes::CONVSP3_OUT_PER_CYCLE, activation::relu8,
            u8::CONVSP3::BIAS_SHIFT, u8::CONVSP3::SHIFT, weights::convsp3_weight, weights::convsp3_bias,
            sizes::CONVSP3_ON_DSPS, sizes::CONVSP3_IN_PER_CYCLE>(convsp2_out, stem_out, height, width);

    // The encoder: double1 at 32 x 32, whose output is also the skip; double2 at 16 x 16.
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, STEM3, DOUBLE1,
                      sizes::DOUBLE1_CONV1::IN_PER_CYCLE, sizes::DOUBLE1_CONV1::OUT_PER_CYCLE, u8::DOUBLE1_CONV1::BIAS_SHIFT,
                      weights::double1_block_0_weight, weights::double1_block_0_bias, sizes::DOUBLE1_CONV1::DSP, sizes::DOUBLE1_CONV1::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE1_CONV1::SHIFT, true, DSP_PACK_SHIFT>(stem_out, double1_relu1, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE1, DOUBLE1,
                      sizes::DOUBLE1_CONV2::IN_PER_CYCLE, sizes::DOUBLE1_CONV2::OUT_PER_CYCLE, u8::DOUBLE1_CONV2::BIAS_SHIFT,
                      weights::double1_block_3_weight, weights::double1_block_3_bias, sizes::DOUBLE1_CONV2::DSP, sizes::DOUBLE1_CONV2::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE1_CONV2::SHIFT, true, DSP_PACK_SHIFT>(double1_relu1, double1_out, height, width);
    duplicate<PATCH_SIZE, PATCH_SIZE, PIXELS>(double1_out, skip, pool1_in, height, width);
    max_pool<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE>(pool1_in, pool1_out, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE1, DOUBLE2,
                      sizes::DOUBLE2_CONV1::IN_PER_CYCLE, sizes::DOUBLE2_CONV1::OUT_PER_CYCLE, u8::DOUBLE2_CONV1::BIAS_SHIFT,
                      weights::double2_block_0_weight, weights::double2_block_0_bias, sizes::DOUBLE2_CONV1::DSP, sizes::DOUBLE2_CONV1::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE2_CONV1::SHIFT, true, DSP_PACK_SHIFT>(pool1_out, double2_relu1, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                      sizes::DOUBLE2_CONV2::IN_PER_CYCLE, sizes::DOUBLE2_CONV2::OUT_PER_CYCLE, u8::DOUBLE2_CONV2::BIAS_SHIFT,
                      weights::double2_block_3_weight, weights::double2_block_3_bias, sizes::DOUBLE2_CONV2::DSP, sizes::DOUBLE2_CONV2::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE2_CONV2::SHIFT, true, DSP_PACK_SHIFT>(double2_relu1, double2_out, height, width);
    // Pool to 8 x 8 and upsample back.
    max_pool<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH>(double2_out, pool2_out, height, width);
    upsample<PATCH_SIZE, PATCH_SIZE, HALF_PATCH / 2, HALF_PATCH / 2>(pool2_out, pool2_up, height, width);

    // The decoder: up1's conv at 16 x 16, double3, upsample to 32 x 32, up2's conv.
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                      sizes::UP1_CONV::IN_PER_CYCLE, sizes::UP1_CONV::OUT_PER_CYCLE, u8::UP1::BIAS_SHIFT,
                      weights::up1_1_weight, weights::up1_1_bias, sizes::UP1_CONV::DSP, sizes::UP1_CONV::TAPS_PER_STEP,
                      activation::relu8, u8::UP1::SHIFT, true, DSP_PACK_SHIFT>(pool2_up, up1_relu, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE3,
                      sizes::DOUBLE3_CONV1::IN_PER_CYCLE, sizes::DOUBLE3_CONV1::OUT_PER_CYCLE, u8::DOUBLE3_CONV1::BIAS_SHIFT,
                      weights::double3_block_0_weight, weights::double3_block_0_bias, sizes::DOUBLE3_CONV1::DSP, sizes::DOUBLE3_CONV1::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE3_CONV1::SHIFT, true, DSP_PACK_SHIFT>(up1_relu, double3_relu1, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE3, DOUBLE3,
                      sizes::DOUBLE3_CONV2::IN_PER_CYCLE, sizes::DOUBLE3_CONV2::OUT_PER_CYCLE, u8::DOUBLE3_CONV2::BIAS_SHIFT,
                      weights::double3_block_3_weight, weights::double3_block_3_bias, sizes::DOUBLE3_CONV2::DSP, sizes::DOUBLE3_CONV2::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE3_CONV2::SHIFT, true, DSP_PACK_SHIFT>(double3_relu1, double3_act, height, width);
    upsample<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH>(double3_act, double3_out, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3, DOUBLE3,
                      sizes::UP2_CONV::IN_PER_CYCLE, sizes::UP2_CONV::OUT_PER_CYCLE, u8::UP2::BIAS_SHIFT,
                      weights::up2_1_weight, weights::up2_1_bias, sizes::UP2_CONV::DSP, sizes::UP2_CONV::TAPS_PER_STEP,
                      activation::relu8, u8::UP2::SHIFT, true, DSP_PACK_SHIFT>(double3_out, up2_relu, height, width);

    // The skip joins (one scale: act8::up2_1 == act8::double1_block_3), then double4 and the
    // final 1x1 conv: the scores.
    concat_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE3, DOUBLE1>(up2_relu, skip, joined, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3 + DOUBLE1, DOUBLE4,
                      sizes::DOUBLE4_CONV1::IN_PER_CYCLE, sizes::DOUBLE4_CONV1::OUT_PER_CYCLE, u8::DOUBLE4_CONV1::BIAS_SHIFT,
                      weights::double4_block_0_weight, weights::double4_block_0_bias, sizes::DOUBLE4_CONV1::DSP, sizes::DOUBLE4_CONV1::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE4_CONV1::SHIFT, true, DSP_PACK_SHIFT>(joined, double4_relu1, height, width);
    conv3x3_tiled_ram<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE4, DOUBLE4,
                      sizes::DOUBLE4_CONV2::IN_PER_CYCLE, sizes::DOUBLE4_CONV2::OUT_PER_CYCLE, u8::DOUBLE4_CONV2::BIAS_SHIFT,
                      weights::double4_block_3_weight, weights::double4_block_3_bias, sizes::DOUBLE4_CONV2::DSP, sizes::DOUBLE4_CONV2::TAPS_PER_STEP,
                      activation::relu8, u8::DOUBLE4_CONV2::SHIFT, true, DSP_PACK_SHIFT>(double4_relu1, double4_out, height, width);
    conv1x1<patch_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS>, DOUBLE4, CLASSES, sizes::FINAL_OUT_PER_CYCLE, activation::none,
            u8::FINAL::BIAS_SHIFT, u8::FINAL::SHIFT, weights::final_weight, weights::final_bias,
            sizes::FINAL_ON_DSPS, sizes::FINAL_IN_PER_CYCLE>(double4_out, scores, height, width);
    store_scores<PATCH_SIZE, PATCH_SIZE, u8::FINAL::SHIFT>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
