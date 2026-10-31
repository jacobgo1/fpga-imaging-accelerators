#include "../../../weights/quantized/justoliunet_bn/justoliunet_bn_int8.hpp"
#include "../justoliunet_opt/justoliunet_opt_formats.hpp"
namespace weights = justoliunet_bn_int8;
namespace model_formats = justoliunet_bn_formats;
#include "../justoliunet_opt_zynq7030/justoliunet_opt_zynq7030.hpp"
#include "../../optimized/shared/pixel_io.hpp"
#include "../../optimized/shared/conv1d_engine.hpp"
#include "../../optimized/shared/conv1d_stream.hpp"
#include "../../optimized/shared/count_cycles.hpp"

namespace shared_formats = justoliunet_opt_formats;

// A layer's sums have its input's fraction bits plus its weights'.
constexpr int CONV1_SUM_FRAC_BITS = shared_formats::IN_FRAC + weights::conv1_weight_frac;
constexpr int CONV2_SUM_FRAC_BITS = model_formats::ACT_FRAC1 + weights::conv2_weight_frac;
constexpr int CONV3_SUM_FRAC_BITS = model_formats::ACT_FRAC2 + weights::conv3_weight_frac;
constexpr int CONV4_SUM_FRAC_BITS = model_formats::ACT_FRAC3 + weights::conv4_weight_frac;
constexpr int DENSE_SUM_FRAC_BITS = model_formats::ACT_FRAC4 + weights::fc_weight_frac;

#include "justoliunet_bn_opt_zynq7030.hpp"

void justoliunet_bn_opt_zynq7030(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
                                 int height, int width, uint32_t& cycles)
{
    // The reader on HP0, the writer on HP1. depth is for co-simulation only.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=1890 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=63 max_write_burst_length=64 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW
    hls::stream<cube_word_t> cube_words[READ_PORTS];
    hls::stream<map_t<int16_t, KEPT_BANDS, 1>> zscores("zscores");
    hls::stream<vec_t<uact_t, CONV1_POSITIONS_PER_CYCLE * 6>> conv1_out("conv1_out");
    hls::stream<vec_t<uact_t, 6>> pool1_out("pool1_out");
    hls::stream<vec_t<uact_t, 12>> pool2_out("pool2_out");
    hls::stream<vec_t<uact_t, 18>> pool3_out("pool3_out");
    hls::stream<vec_t<uact_t, 24>> pool4_out("pool4_out");
    hls::stream<vec_t<acc_t, CLASSES>> scores("scores");
    hls::stream<bool> all_written("all_written");
    // A layer reads a pixel's first positions back to back to fill its window:
    // room for them, so the layer before does not wait.
    #pragma HLS STREAM variable=cube_words depth=64
    #pragma HLS STREAM variable=pool1_out  depth=8
    #pragma HLS STREAM variable=pool2_out  depth=8
    #pragma HLS STREAM variable=pool3_out  depth=8
    #pragma HLS STREAM variable=scores     depth=8

    read_pixels_ahead<WORDS_PER_PIXEL, BLOCK_PIXELS, READ_PORTS, 0>(din0, cube_words[0], height, width);
    zscore_pixels<WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD, BLOCK_PIXELS, READ_PORTS,
                  shared_formats::PREP_FRAC, shared_formats::IN_MAX, shared_formats::A, shared_formats::B,
                  ZSCORE_ON_DSPS>(cube_words, zscores, height, width);

    // conv1: 1 -> 6, 102 outputs, in LUTs (the 12-bit input cannot be packed).
    conv1d_engine<KEPT_BANDS, 1, 6, KERNEL_SIZE, 102, CONV1_POSITIONS_PER_CYCLE, 6,
                  CONV1_SUM_FRAC_BITS - weights::conv1_bias_frac, CONV1_SUM_FRAC_BITS - model_formats::ACT_FRAC1,
                  shared_formats::ACT_MAX, weights::conv1_weight, weights::conv1_bias, false, false>(
        zscores, conv1_out, height, width);
    pool_adjacent<102 / CONV1_POSITIONS_PER_CYCLE, 6>(conv1_out, pool1_out, height, width);

    // conv2: 6 -> 12, 46 outputs; 51 positions in, 23 out.
    conv1d_stream<51, 6, 12, KERNEL_SIZE, 46, CONV2_OUT_PER_CYCLE, CONV2_IN_PER_CYCLE,
                  CONV2_SUM_FRAC_BITS - weights::conv2_bias_frac, CONV2_SUM_FRAC_BITS - model_formats::ACT_FRAC2,
                  shared_formats::ACT_MAX, weights::conv2_weight, weights::conv2_bias, CONV2_PACKED, CONV2_PACKED>(
        pool1_out, pool2_out, height, width);

    // conv3: 12 -> 18, 18 outputs; 23 in, 9 out.
    conv1d_stream<23, 12, 18, KERNEL_SIZE, 18, CONV3_OUT_PER_CYCLE, CONV3_IN_PER_CYCLE,
                  CONV3_SUM_FRAC_BITS - weights::conv3_bias_frac, CONV3_SUM_FRAC_BITS - model_formats::ACT_FRAC3,
                  shared_formats::ACT_MAX, weights::conv3_weight, weights::conv3_bias, CONV3_PACKED, CONV3_PACKED>(
        pool2_out, pool3_out, height, width);

    // conv4: 18 -> 24, 4 outputs; 9 in, 2 out.
    conv1d_stream<9, 18, 24, KERNEL_SIZE, 4, CONV4_OUT_PER_CYCLE, CONV4_IN_PER_CYCLE,
                  CONV4_SUM_FRAC_BITS - weights::conv4_bias_frac, CONV4_SUM_FRAC_BITS - model_formats::ACT_FRAC4,
                  shared_formats::ACT_MAX, weights::conv4_weight, weights::conv4_bias, CONV4_PACKED, CONV4_PACKED>(
        pool3_out, pool4_out, height, width);

    // flatten and dense: 2 positions x 24 -> 3, the scores.
    dense_stream<2, 24, CLASSES, DENSE_FEATURES_PER_CYCLE, DENSE_SUM_FRAC_BITS - weights::fc_bias_frac,
                 shared_formats::OUT_FRAC - DENSE_SUM_FRAC_BITS, weights::fc_weight, weights::fc_bias, false>(
        pool4_out, scores, height, width);
    store_pixels<CLASSES, 64, 8>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
