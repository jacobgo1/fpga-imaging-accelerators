#include "../../../../weights/quantized/justoliunet/justoliunet_int8.hpp"
#include "../justoliunet_opt/justoliunet_opt_formats.hpp"
namespace weights = justoliunet_int8;
namespace model_formats = justoliunet_formats;
#include "justoliunet_opt_zynq7020.hpp"
#include "../../../optimized/pixel_io.hpp"
#include "../../../optimized/conv1d.hpp"
#include "../../../optimized/count_cycles.hpp"

namespace shared_formats = justoliunet_opt_formats;

// A layer's sums have its input's fraction bits plus its weights'.
constexpr int CONV1_SUM_FRAC_BITS = shared_formats::IN_FRAC + weights::conv1_weight_frac;
constexpr int CONV2_SUM_FRAC_BITS = model_formats::ACT_FRAC1 + weights::conv2_weight_frac;
constexpr int CONV3_SUM_FRAC_BITS = model_formats::ACT_FRAC2 + weights::conv3_weight_frac;
constexpr int CONV4_SUM_FRAC_BITS = model_formats::ACT_FRAC3 + weights::conv4_weight_frac;
constexpr int DENSE_SUM_FRAC_BITS = model_formats::ACT_FRAC4 + weights::fc_weight_frac;

void justoliunet_opt_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<score_word_t> dout,
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
    // Each layer: its windows, the conv's outputs a window, the pooled positions.
    hls::stream<map_t<int16_t, CONV1_POSITIONS_PER_CYCLE + KERNEL_SIZE - 1, 1>> conv1_windows("conv1_windows");
    hls::stream<map_t<uact_t, CONV1_POSITIONS_PER_CYCLE, 6>> conv1_out("conv1_out");
    // A windows process reads a pixel's first positions back to back to fill its
    // window: room for them, so the layer before does not wait.
    hls::stream<vec_t<uact_t, 6>, 8> pool1_out("pool1_out");
    hls::stream<map_t<uact_t, KERNEL_SIZE, 6>> conv2_windows("conv2_windows");
    hls::stream<map_t<uact_t, 1, 12>> conv2_out("conv2_out");
    hls::stream<vec_t<uact_t, 12>, 8> pool2_out("pool2_out");
    hls::stream<map_t<uact_t, KERNEL_SIZE, 12>> conv3_windows("conv3_windows");
    hls::stream<map_t<uact_t, 1, 18>> conv3_out("conv3_out");
    hls::stream<vec_t<uact_t, 18>, 8> pool3_out("pool3_out");
    hls::stream<map_t<uact_t, KERNEL_SIZE, 18>> conv4_windows("conv4_windows");
    hls::stream<map_t<uact_t, 1, 24>> conv4_out("conv4_out");
    hls::stream<vec_t<uact_t, 24>> pool4_out("pool4_out");
    hls::stream<vec_t<sum_t, CLASSES>, 8> scores("scores");
    hls::stream<bool> all_written("all_written");
    #pragma HLS STREAM variable=cube_words depth=64

    read_pixels<WORDS_PER_PIXEL, BLOCK_PIXELS, READ_PORTS, 0>(din0, cube_words[0], height, width);
    zscore_pixels<WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD, BLOCK_PIXELS, READ_PORTS,
                  shared_formats::PREP_FRAC, shared_formats::IN_MAX, shared_formats::A, shared_formats::B,
                  ZSCORE_ON_DSPS>(cube_words, zscores, height, width);

    // conv1: 1 -> 6, 102 outputs, in LUTs (the 12-bit input cannot be packed).
    map_windows<KEPT_BANDS, 1, KERNEL_SIZE, 102, CONV1_POSITIONS_PER_CYCLE>(zscores, conv1_windows, height, width);
    conv1d<1, 6, KERNEL_SIZE, 102, CONV1_POSITIONS_PER_CYCLE, 6, 1,
           CONV1_SUM_FRAC_BITS - weights::conv1_bias_frac, CONV1_SUM_FRAC_BITS - model_formats::ACT_FRAC1,
           shared_formats::ACT_MAX, weights::conv1_weight, weights::conv1_bias, false, false, DSP_PACK_SHIFT>(
        conv1_windows, conv1_out, height, width);
    pool1d<102, CONV1_POSITIONS_PER_CYCLE, 6>(conv1_out, pool1_out, height, width);

    // conv2: 6 -> 12, 46 outputs; 51 positions in, 23 out.
    position_windows<51, 6, KERNEL_SIZE, 46>(pool1_out, conv2_windows, height, width);
    conv1d<6, 12, KERNEL_SIZE, 46, 1, CONV2_OUT_PER_CYCLE, CONV2_IN_PER_CYCLE,
           CONV2_SUM_FRAC_BITS - weights::conv2_bias_frac, CONV2_SUM_FRAC_BITS - model_formats::ACT_FRAC2,
           shared_formats::ACT_MAX, weights::conv2_weight, weights::conv2_bias, CONV2_PACKED, CONV2_PACKED, DSP_PACK_SHIFT>(
        conv2_windows, conv2_out, height, width);
    pool1d<46, 1, 12>(conv2_out, pool2_out, height, width);

    // conv3: 12 -> 18, 18 outputs; 23 in, 9 out.
    position_windows<23, 12, KERNEL_SIZE, 18>(pool2_out, conv3_windows, height, width);
    conv1d<12, 18, KERNEL_SIZE, 18, 1, CONV3_OUT_PER_CYCLE, CONV3_IN_PER_CYCLE,
           CONV3_SUM_FRAC_BITS - weights::conv3_bias_frac, CONV3_SUM_FRAC_BITS - model_formats::ACT_FRAC3,
           shared_formats::ACT_MAX, weights::conv3_weight, weights::conv3_bias, CONV3_PACKED, CONV3_PACKED, DSP_PACK_SHIFT>(
        conv3_windows, conv3_out, height, width);
    pool1d<18, 1, 18>(conv3_out, pool3_out, height, width);

    // conv4: 18 -> 24, 4 outputs; 9 in, 2 out.
    position_windows<9, 18, KERNEL_SIZE, 4>(pool3_out, conv4_windows, height, width);
    conv1d<18, 24, KERNEL_SIZE, 4, 1, CONV4_OUT_PER_CYCLE, CONV4_IN_PER_CYCLE,
           CONV4_SUM_FRAC_BITS - weights::conv4_bias_frac, CONV4_SUM_FRAC_BITS - model_formats::ACT_FRAC4,
           shared_formats::ACT_MAX, weights::conv4_weight, weights::conv4_bias, CONV4_PACKED, CONV4_PACKED, DSP_PACK_SHIFT>(
        conv4_windows, conv4_out, height, width);
    pool1d<4, 1, 24>(conv4_out, pool4_out, height, width);

    // flatten and dense: 2 positions x 24 -> 3, the scores.
    dense_positions<2, 24, CLASSES, DENSE_FEATURES_PER_CYCLE, DENSE_SUM_FRAC_BITS - weights::fc_bias_frac,
                    shared_formats::OUT_FRAC - DENSE_SUM_FRAC_BITS, weights::fc_weight, weights::fc_bias, false>(
        pool4_out, scores, height, width);
    store_pixels<CLASSES, 64, 8>(scores, dout, height, width, all_written);

    count_cycles(all_written, cycles);   // last: C simulation runs the processes in order
}
