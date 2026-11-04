#include "sp_unet_medium_opt.hpp"
#include "../../../../optimized/read_patch_rows_simple.hpp"
#include "../../../../optimized/pixel_buffer.hpp"
#include "../../../../optimized/pointwise_packed.hpp"
#include "../../../../optimized/conv1x1.hpp"
#include "../../../../optimized/conv3x3_tiled.hpp"
#include "../../../../optimized/max_pool.hpp"
#include "../../../../optimized/duplicate.hpp"
#include "../../../../optimized/upsample.hpp"
#include "../../../../optimized/concat_pixels.hpp"
#include "../../../../optimized/store_scores.hpp"
#include "../../../../optimized/count_cycles.hpp"

#include "../../../../../weights/quantized/sp_unet_medium/sp_unet_medium_int8.hpp"
namespace weights = sp_unet_medium_int8;

#include "../../Small/sp_unet_small_opt/sp_unet_opt_common.hpp"

void sp_unet_medium_opt(const cube_word_t* din0, const cube_word_t* din1,
                        const cube_word_t* din2, const cube_word_t* din3,
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
    constexpr int PIXELS = PATCH_SIZE * PATCH_SIZE;
    hls::stream<cube_word_t> cube_words[READ_PORTS];
    hls::stream<vec_t<act8_t, sizes::CONVSP1_BANDS_PER_CYCLE>, 64> zscores("zscores");
    // 16 deep between the layers, not 2: they run at the same rate, and a pipeline
    // freezes all its stages when one read or write waits, so one conv's short
    // wait froze the next (11 503 cycles a patch with 2, 9 856 with 16).
    hls::stream<vec_t<act_t, STEM1>, 16> convsp1_out("convsp1_out");
    hls::stream<vec_t<act_t, STEM2>, 16> convsp2_out("convsp2_out");
    hls::stream<vec_t<act_t, STEM3>, 16> stem_out("stem_out");
    hls::stream<vec_t<act_t, DOUBLE1>, 16> double1_relu1("double1_relu1"), double1_out("double1_out");
    // The skip waits up to a patch for up2; too shallow and it deadlocks.
    hls::stream<vec_t<act_t, DOUBLE1>, 2048> skip("skip");
    // The pooled pixels and the upsampled rows come in bursts: room for a burst.
    hls::stream<vec_t<act_t, DOUBLE1>, 16> pool1_in("pool1_in");
    hls::stream<vec_t<act_t, DOUBLE1>, 16> pool1_out("pool1_out");
    hls::stream<vec_t<act_t, DOUBLE2>, 16> double2_relu1("double2_relu1"), double2_out("double2_out");
    hls::stream<vec_t<act_t, DOUBLE2>, 16> pool2_out("pool2_out");
    hls::stream<vec_t<act_t, DOUBLE2>, 32> pool2_up("pool2_up");
    hls::stream<vec_t<act_t, DOUBLE2>, 16> up1_relu("up1_relu");
    hls::stream<vec_t<act_t, DOUBLE3>, 16> double3_relu1("double3_relu1"), double3_act("double3_act");
    hls::stream<vec_t<act_t, DOUBLE3>, 64> double3_out("double3_out");
    hls::stream<vec_t<act_t, DOUBLE3>, 16> up2_relu("up2_relu");
    hls::stream<vec_t<act_t, DOUBLE3 + DOUBLE1>, 16> joined("joined");
    hls::stream<vec_t<act_t, DOUBLE4>, 16> double4_relu1("double4_relu1"), double4_out("double4_out");
    hls::stream<vec_t<sum_t, CLASSES>, 32> scores("scores");
    hls::stream<bool> all_written("all_written");

    // pixel_buffer takes a strip's words in 480 of each 896 steps: room for a
    // strip row lets the readers use the other 416 too (11 503 to 9 856 cycles a patch).
    #pragma HLS STREAM variable=cube_words depth=512
    #pragma HLS BIND_STORAGE variable=cube_words type=fifo impl=bram
    #pragma HLS BIND_STORAGE variable=skip type=fifo impl=uram
    #pragma HLS BIND_STORAGE variable=convsp1_out   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=convsp2_out   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stem_out      type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double1_relu1 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double1_out   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=pool1_in      type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=pool1_out     type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double2_relu1 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double2_out   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=pool2_out     type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=pool2_up      type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=up1_relu      type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double3_relu1 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double3_act   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double3_out   type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=up2_relu      type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=joined        type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double4_relu1 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=double4_out   type=fifo impl=srl

    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0>(din0, cube_words[0], height, width);
    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 1>(din1, cube_words[1], height, width);
    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 2>(din2, cube_words[2], height, width);
    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 3>(din3, cube_words[3], height, width);
    pixel_buffer<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD,
                 sizes::CONVSP1_BANDS_PER_CYCLE, READ_PORTS, CONVSP1_CHUNKS, zscore::FRAC, zscore::IN_MAX,
                 zscore::A, zscore::B, sizes::ZSCORE_ON_DSPS>(cube_words, zscores, height, width);

    // The spectral stem, 1x1 convs; convsp1 two multiplies per DSP.
    pointwise_packed<PATCH_SIZE, PATCH_SIZE, PIXELS, MODEL_BANDS, STEM1, sizes::CONVSP1_BANDS_PER_CYCLE,
                     CONVSP1_FRACS::BIAS_SHIFT, CONVSP1_FRACS::SHIFT, weights::convsp1_weight, weights::convsp1_bias,
                     DSP_PACK_SHIFT>(zscores, convsp1_out, height, width);
    conv1x1<patch_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS>, STEM1, STEM2, sizes::CONVSP2_OUT_PER_CYCLE, activation::relu,
            CONVSP2_FRACS::BIAS_SHIFT, CONVSP2_FRACS::SHIFT, weights::convsp2_weight, weights::convsp2_bias,
            sizes::CONVSP2_3_ON_DSPS>(convsp1_out, convsp2_out, height, width);
    conv1x1<patch_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS>, STEM2, STEM3, sizes::CONVSP3_OUT_PER_CYCLE, activation::relu,
            CONVSP3_FRACS::BIAS_SHIFT, CONVSP3_FRACS::SHIFT, weights::convsp3_weight, weights::convsp3_bias,
            sizes::CONVSP2_3_ON_DSPS>(convsp2_out, stem_out, height, width);

    // The encoder: double1 at 32 x 32, whose output is also the skip; double2 at 16 x 16.
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, STEM3, DOUBLE1,
                  sizes::DOUBLE1_CONV1::IN_PER_CYCLE, sizes::DOUBLE1_CONV1::OUT_PER_CYCLE, DOUBLE1_CONV1_FRACS::BIAS_SHIFT,
                  weights::double1_block_0_weight, weights::double1_block_0_bias, sizes::DOUBLE1_CONV1::DSP,
                  activation::relu, DOUBLE1_CONV1_FRACS::SHIFT>(stem_out, double1_relu1, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE1, DOUBLE1,
                  sizes::DOUBLE1_CONV2::IN_PER_CYCLE, sizes::DOUBLE1_CONV2::OUT_PER_CYCLE, DOUBLE1_CONV2_FRACS::BIAS_SHIFT,
                  weights::double1_block_3_weight, weights::double1_block_3_bias, sizes::DOUBLE1_CONV2::DSP,
                  activation::relu, DOUBLE1_CONV2_FRACS::SHIFT>(double1_relu1, double1_out, height, width);
    duplicate<PATCH_SIZE, PATCH_SIZE, PIXELS>(double1_out, skip, pool1_in, height, width);
    max_pool<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE>(pool1_in, pool1_out, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE1, DOUBLE2,
                  sizes::DOUBLE2_CONV1::IN_PER_CYCLE, sizes::DOUBLE2_CONV1::OUT_PER_CYCLE, DOUBLE2_CONV1_FRACS::BIAS_SHIFT,
                  weights::double2_block_0_weight, weights::double2_block_0_bias, sizes::DOUBLE2_CONV1::DSP,
                  activation::relu, DOUBLE2_CONV1_FRACS::SHIFT>(pool1_out, double2_relu1, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                  sizes::DOUBLE2_CONV2::IN_PER_CYCLE, sizes::DOUBLE2_CONV2::OUT_PER_CYCLE, DOUBLE2_CONV2_FRACS::BIAS_SHIFT,
                  weights::double2_block_3_weight, weights::double2_block_3_bias, sizes::DOUBLE2_CONV2::DSP,
                  activation::relu, DOUBLE2_CONV2_FRACS::SHIFT>(double2_relu1, double2_out, height, width);
    // Pool to 8 x 8 and upsample back.
    max_pool<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH>(double2_out, pool2_out, height, width);
    upsample<PATCH_SIZE, PATCH_SIZE, HALF_PATCH / 2, HALF_PATCH / 2>(pool2_out, pool2_up, height, width);

    // The decoder: up1's conv at 16 x 16, double3, upsample to 32 x 32, up2's conv.
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE2,
                  sizes::UP1_CONV::IN_PER_CYCLE, sizes::UP1_CONV::OUT_PER_CYCLE, UP1_FRACS::BIAS_SHIFT,
                  weights::up1_1_weight, weights::up1_1_bias, sizes::UP1_CONV::DSP,
                  activation::relu, UP1_FRACS::SHIFT>(pool2_up, up1_relu, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE2, DOUBLE3,
                  sizes::DOUBLE3_CONV1::IN_PER_CYCLE, sizes::DOUBLE3_CONV1::OUT_PER_CYCLE, DOUBLE3_CONV1_FRACS::BIAS_SHIFT,
                  weights::double3_block_0_weight, weights::double3_block_0_bias, sizes::DOUBLE3_CONV1::DSP,
                  activation::relu, DOUBLE3_CONV1_FRACS::SHIFT>(up1_relu, double3_relu1, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, DOUBLE3, DOUBLE3,
                  sizes::DOUBLE3_CONV2::IN_PER_CYCLE, sizes::DOUBLE3_CONV2::OUT_PER_CYCLE, DOUBLE3_CONV2_FRACS::BIAS_SHIFT,
                  weights::double3_block_3_weight, weights::double3_block_3_bias, sizes::DOUBLE3_CONV2::DSP,
                  activation::relu, DOUBLE3_CONV2_FRACS::SHIFT>(double3_relu1, double3_act, height, width);
    upsample<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH>(double3_act, double3_out, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3, DOUBLE3,
                  sizes::UP2_CONV::IN_PER_CYCLE, sizes::UP2_CONV::OUT_PER_CYCLE, UP2_FRACS::BIAS_SHIFT,
                  weights::up2_1_weight, weights::up2_1_bias, sizes::UP2_CONV::DSP,
                  activation::relu, UP2_FRACS::SHIFT>(double3_out, up2_relu, height, width);

    // The skip joins, then double4 and the final 1x1 conv: the scores.
    concat_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS, DOUBLE3, DOUBLE1>(up2_relu, skip, joined, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE3 + DOUBLE1, DOUBLE4,
                  sizes::DOUBLE4_CONV1::IN_PER_CYCLE, sizes::DOUBLE4_CONV1::OUT_PER_CYCLE, DOUBLE4_CONV1_FRACS::BIAS_SHIFT,
                  weights::double4_block_0_weight, weights::double4_block_0_bias, sizes::DOUBLE4_CONV1::DSP,
                  activation::relu, DOUBLE4_CONV1_FRACS::SHIFT>(joined, double4_relu1, height, width);
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DOUBLE4, DOUBLE4,
                  sizes::DOUBLE4_CONV2::IN_PER_CYCLE, sizes::DOUBLE4_CONV2::OUT_PER_CYCLE, DOUBLE4_CONV2_FRACS::BIAS_SHIFT,
                  weights::double4_block_3_weight, weights::double4_block_3_bias, sizes::DOUBLE4_CONV2::DSP,
                  activation::relu, DOUBLE4_CONV2_FRACS::SHIFT>(double4_relu1, double4_out, height, width);
    conv1x1<patch_pixels<PATCH_SIZE, PATCH_SIZE, PIXELS>, DOUBLE4, CLASSES, sizes::FINAL_OUT_PER_CYCLE, activation::none,
            FINAL_FRACS::BIAS_SHIFT, FINAL_FRACS::SHIFT, weights::final_weight, weights::final_bias, false>(
        double4_out, scores, height, width);
    store_scores<PATCH_SIZE, PATCH_SIZE, FINAL_FRACS::SHIFT>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
