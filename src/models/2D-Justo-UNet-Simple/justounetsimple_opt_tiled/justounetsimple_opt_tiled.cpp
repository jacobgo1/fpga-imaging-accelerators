#include "justounetsimple_opt_tiled.hpp"
#include "../../../optimized/read_patch_rows_simple.hpp"
#include "../../../optimized/pixel_buffer.hpp"
#include "../../../optimized/conv3x3_tiled_chunks.hpp"
#include "../../../optimized/conv3x3_tiled.hpp"
#include "../../../optimized/max_pool.hpp"
#include "../../../optimized/upsample.hpp"
#include "../../../optimized/store_scores.hpp"
#include "../../../optimized/count_cycles.hpp"

#include "../../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace weights = justounetsimple_int8;

// A layer's sums have the fraction bits of its input plus those of its weights.
// The bias is shifted up to them, the sums down to the next layer's activations.
constexpr int CONV1_SUM_FRAC_BITS = ZSCORE_FRAC_BITS + weights::conv1_weight_frac;
constexpr int CONV2_SUM_FRAC_BITS = ACTIVATION_FRAC_BITS + weights::conv2_weight_frac;
constexpr int CONV3_SUM_FRAC_BITS = ACTIVATION_FRAC_BITS + weights::conv3_weight_frac;
constexpr int CONV4_SUM_FRAC_BITS = ACTIVATION_FRAC_BITS + weights::conv4_weight_frac;

constexpr int CONV1_BIAS_SHIFT = CONV1_SUM_FRAC_BITS - weights::conv1_bias_frac;
constexpr int CONV2_BIAS_SHIFT = CONV2_SUM_FRAC_BITS - weights::conv2_bias_frac;
constexpr int CONV3_BIAS_SHIFT = CONV3_SUM_FRAC_BITS - weights::conv3_bias_frac;
constexpr int CONV4_BIAS_SHIFT = CONV4_SUM_FRAC_BITS - weights::conv4_bias_frac;

constexpr int CONV1_OUTPUT_SHIFT = CONV1_SUM_FRAC_BITS - ACTIVATION_FRAC_BITS;
constexpr int CONV2_OUTPUT_SHIFT = CONV2_SUM_FRAC_BITS - ACTIVATION_FRAC_BITS;
constexpr int CONV3_OUTPUT_SHIFT = CONV3_SUM_FRAC_BITS - ACTIVATION_FRAC_BITS;
constexpr int CONV4_OUTPUT_SHIFT = CONV4_SUM_FRAC_BITS - SCORE_FRAC_BITS;

void justounetsimple_opt_tiled(const cube_word_t* din0, const cube_word_t* din1,
                               const cube_word_t* din2, const cube_word_t* din3,
                               hls::burst_maxi<score_word_t> dout, int height, int width, uint32_t& cycles)
{
    // Four readers on HP0-3, the writer on HPC0. depth is for co-simulation only.
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

    // The FIFOs between the processes, each with its depth. The strip buffer holds
    // only a strip, not a patch as the patch buffer did, so the readers' FIFOs take
    // up their bursts' jitter: 512 deep in block RAM, as in sp_unet_small_opt.
    hls::stream<cube_word_t> cube_words[READ_PORTS];
    #pragma HLS STREAM variable=cube_words depth=512
    #pragma HLS BIND_STORAGE variable=cube_words type=fifo impl=bram
    hls::stream<vec_t<act8_t, CONV1_BANDS_PER_CYCLE>, 64> zscores("zscores");
    hls::stream<vec_t<act_t, BASE_CHANNELS>, 16> conv1_act("conv1_act");
    hls::stream<vec_t<act_t, BASE_CHANNELS>, 256> conv1_pooled("conv1_pooled");
    hls::stream<vec_t<act_t, 2 * BASE_CHANNELS>, 16> conv2_act("conv2_act");
    hls::stream<vec_t<act_t, 2 * BASE_CHANNELS>, 16> conv2_pooled("conv2_pooled");
    hls::stream<vec_t<act_t, 2 * BASE_CHANNELS>, 32> conv2_upsampled("conv2_upsampled");
    hls::stream<vec_t<act_t, BASE_CHANNELS>, 16> conv3_act("conv3_act");
    hls::stream<vec_t<act_t, BASE_CHANNELS>, 64> conv3_upsampled("conv3_upsampled");
    hls::stream<vec_t<sum_t, CLASSES>, 32> scores("scores");
    hls::stream<bool> all_written("all_written");

    // The model layer by layer.
    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0>(din0, cube_words[0], height, width);
    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 1>(din1, cube_words[1], height, width);
    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 2>(din2, cube_words[2], height, width);
    read_patch_rows_simple<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 3>(din3, cube_words[3], height, width);
    pixel_buffer<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD, CONV1_BANDS_PER_CYCLE,
                 READ_PORTS, CONV1_CHUNKS, zscore::FRAC, zscore::IN_MAX, zscore::A, zscore::B, ZSCORE_ON_DSPS>(
        cube_words, zscores, height, width);

    // conv1: 110 -> 6 at 32 x 32, a pixel's 24-band chunks in, two multiplies per DSP; ReLU; pool.
    conv3x3_tiled_chunks<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, MODEL_BANDS, BASE_CHANNELS,
                         CONV1_BANDS_PER_CYCLE, BASE_CHANNELS, CONV1_BIAS_SHIFT, weights::conv1_weight,
                         weights::conv1_bias, DSP_PACK_SHIFT, activation::relu, CONV1_OUTPUT_SHIFT>(
        zscores, conv1_act, height, width);
    max_pool<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE>(conv1_act, conv1_pooled, height, width);

    // conv2: 6 -> 12 at 16 x 16; ReLU; pool to 8 x 8 and upsample back.
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, BASE_CHANNELS, 2 * BASE_CHANNELS, BASE_CHANNELS, 1,
                  CONV2_BIAS_SHIFT, weights::conv2_weight, weights::conv2_bias, CONV2_TO_4_ON_DSPS, activation::relu,
                  CONV2_OUTPUT_SHIFT>(conv1_pooled, conv2_act, height, width);
    max_pool<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH>(conv2_act, conv2_pooled, height, width);
    upsample<PATCH_SIZE, PATCH_SIZE, HALF_PATCH / 2, HALF_PATCH / 2>(conv2_pooled, conv2_upsampled, height, width);

    // conv3: 12 -> 6 at 16 x 16; ReLU; upsample to 32 x 32.
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, 2 * BASE_CHANNELS, BASE_CHANNELS, 2 * BASE_CHANNELS, 1,
                  CONV3_BIAS_SHIFT, weights::conv3_weight, weights::conv3_bias, CONV2_TO_4_ON_DSPS, activation::relu,
                  CONV3_OUTPUT_SHIFT>(conv2_upsampled, conv3_act, height, width);
    upsample<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH>(conv3_act, conv3_upsampled, height, width);

    // conv4: 6 -> 3 at 32 x 32, the scores.
    conv3x3_tiled<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, BASE_CHANNELS, CLASSES, BASE_CHANNELS, 1,
                  CONV4_BIAS_SHIFT, weights::conv4_weight, weights::conv4_bias, CONV2_TO_4_ON_DSPS>(
        conv3_upsampled, scores, height, width);
    store_scores<PATCH_SIZE, PATCH_SIZE, CONV4_OUTPUT_SHIFT>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
