#include "justounetsimple_opt.hpp"
#include "../../optimized/shared/zscore_int8.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/zcu104/patch_buffer.hpp"
#include "../../optimized/zcu104/conv3x3_wide.hpp"
#include "../../optimized/shared/conv3x3_narrow.hpp"
#include "../../optimized/shared/relu_pool.hpp"
#include "../../optimized/shared/relu_pool_up.hpp"
#include "../../optimized/shared/relu_up.hpp"
#include "../../optimized/shared/store_scores.hpp"
#include "../../optimized/shared/count_cycles.hpp"
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace weights = justounetsimple_int8;
namespace zscore = zscore_int8;

constexpr int HALF_PATCH = PATCH_SIZE / 2;
constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = zscore::FIRST_BAND / BANDS_PER_WORD;
constexpr bool ZSCORE_ON_DSPS = true;
constexpr bool CONV2_TO_4_ON_DSPS = true;

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

void justounetsimple_opt(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                         hls::burst_maxi<cube_word_t> din2, hls::burst_maxi<cube_word_t> din3,
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

    hls::stream<cube_word_t> cube_words[READ_PORTS];
    hls::stream<vec_t<act8_t, CONV1_BANDS_PER_CYCLE>> zscores("zscores");
    hls::stream<vec_t<acc_t, BASE_CHANNELS>> conv1_sums("conv1_sums");
    hls::stream<vec_t<act_t, BASE_CHANNELS>> conv1_pooled("conv1_pooled");
    hls::stream<vec_t<acc_t, 2 * BASE_CHANNELS>> conv2_sums("conv2_sums");
    hls::stream<vec_t<act_t, 2 * BASE_CHANNELS>> conv2_pooled_upsampled("conv2_pooled_upsampled");
    hls::stream<vec_t<acc_t, BASE_CHANNELS>> conv3_sums("conv3_sums");
    hls::stream<vec_t<act_t, BASE_CHANNELS>> conv3_upsampled("conv3_upsampled");
    hls::stream<vec_t<acc_t, CLASSES>> conv4_sums("conv4_sums");
    hls::stream<bool> all_written("all_written");

    // Deep enough that no layer waits for a slower one's bursts.
    #pragma HLS STREAM variable=cube_words             depth=64
    #pragma HLS STREAM variable=zscores                depth=64
    #pragma HLS STREAM variable=conv1_pooled           depth=256
    #pragma HLS STREAM variable=conv2_pooled_upsampled depth=32
    #pragma HLS STREAM variable=conv3_upsampled        depth=64
    #pragma HLS STREAM variable=conv4_sums             depth=32

    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0>(din0, cube_words[0], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 1>(din1, cube_words[1], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 2>(din2, cube_words[2], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 3>(din3, cube_words[3], height, width);
    patch_buffer<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD, CONV1_BANDS_PER_CYCLE,
                 READ_PORTS, zscore::FRAC, zscore::IN_MAX, zscore::A, zscore::B, ZSCORE_ON_DSPS>(
        cube_words, zscores, height, width);

    // conv1: 110 -> 6 at 32 x 32, 32 bands a cycle, two multiplies per DSP.
    conv3x3_wide<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, MODEL_BANDS, BASE_CHANNELS, CONV1_BANDS_PER_CYCLE,
                 CONV1_BIAS_SHIFT, weights::conv1_weight, weights::conv1_bias>(zscores, conv1_sums, height, width);
    relu_pool<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, CONV1_OUTPUT_SHIFT>(
        conv1_sums, conv1_pooled, height, width);

    // conv2: 6 -> 12 at 16 x 16.
    conv3x3_narrow<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, BASE_CHANNELS, 2 * BASE_CHANNELS, CONV2_BIAS_SHIFT,
                   weights::conv2_weight, weights::conv2_bias, CONV2_TO_4_ON_DSPS>(
        conv1_pooled, conv2_sums, height, width);
    relu_pool_up<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, CONV2_OUTPUT_SHIFT>(
        conv2_sums, conv2_pooled_upsampled, height, width);

    // conv3: 12 -> 6 at 16 x 16.
    conv3x3_narrow<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, 2 * BASE_CHANNELS, BASE_CHANNELS, CONV3_BIAS_SHIFT,
                   weights::conv3_weight, weights::conv3_bias, CONV2_TO_4_ON_DSPS>(
        conv2_pooled_upsampled, conv3_sums, height, width);
    relu_up<PATCH_SIZE, PATCH_SIZE, HALF_PATCH, HALF_PATCH, CONV3_OUTPUT_SHIFT>(
        conv3_sums, conv3_upsampled, height, width);

    // conv4: 6 -> 3 at 32 x 32, the scores.
    conv3x3_narrow<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, BASE_CHANNELS, CLASSES, CONV4_BIAS_SHIFT,
                   weights::conv4_weight, weights::conv4_bias, CONV2_TO_4_ON_DSPS>(
        conv3_upsampled, conv4_sums, height, width);
    store_scores<PATCH_SIZE, PATCH_SIZE, CONV4_OUTPUT_SHIFT>(conv4_sums, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
