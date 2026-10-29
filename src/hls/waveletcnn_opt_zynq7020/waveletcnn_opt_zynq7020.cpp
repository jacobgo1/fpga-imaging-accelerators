#include "waveletcnn_opt_zynq7020.hpp"
#include "../waveletcnn_opt/waveletcnn_prep.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/pixel_buffer.hpp"
#include "../../optimized/shared/dwt_proj.hpp"
#include "../../optimized/shared/conv3x3_serial.hpp"
#include "../../optimized/shared/leaky_head.hpp"
#include "../../optimized/shared/store_scores.hpp"
#include "../../../weights/quantized/waveletcnn/waveletcnn_int8.hpp"

namespace weights = waveletcnn_int8;
namespace zscore = waveletcnn_prep;

constexpr int WORDS_PER_PIXEL = CUBE_BANDS / BANDS_PER_WORD;
constexpr int FIRST_KEPT_WORD = zscore::FIRST_BAND / BANDS_PER_WORD;
constexpr int PIXELS_PER_BURST = 8, BURSTS_IN_FLIGHT = 2;
constexpr bool ZSCORE_ON_DSPS = false;

// A layer's sums have its input's fraction bits plus its weights'. The DWT's
// products keep all of theirs (exact in int16).
constexpr int DWT_FRAC_BITS = ZSCORE_FRAC_BITS + weights::dwt_spectral_h0_frac;
constexpr int PROJECTION_SUM_FRAC_BITS = DWT_FRAC_BITS + weights::projection_weight_frac;
constexpr int CONV_SUM_FRAC_BITS = ACTIVATION_FRAC_BITS + weights::conv_block_0_weight_frac;
constexpr int HEAD_SUM_FRAC_BITS = ACTIVATION_FRAC_BITS + weights::seg_head_0_weight_frac;

void waveletcnn_opt_zynq7020(hls::burst_maxi<cube_word_t> din0, hls::burst_maxi<cube_word_t> din1,
                        hls::burst_maxi<score_word_t> dout, int height, int width)
{
    // Two readers on HP0-1, the writer on HP2. depth is for co-simulation only.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=2
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=2
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem2 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<cube_word_t> cube_words[READ_PORTS];
    hls::stream<vec_t<act8_t, DWT_BANDS_PER_CYCLE>> zscores("zscores");
    hls::stream<vec_t<act_t, MID_CHANNELS>> projected("projected");
    hls::stream<vec_t<acc_t, MID_CHANNELS>> conv_sums("conv_sums");
    hls::stream<vec_t<acc_t, CLASSES>> scores("scores");
    #pragma HLS STREAM variable=cube_words depth=64
    #pragma HLS STREAM variable=zscores    depth=64
    #pragma HLS STREAM variable=projected  depth=64
    #pragma HLS STREAM variable=conv_sums  depth=32
    #pragma HLS STREAM variable=scores     depth=32

    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 0, PIXELS_PER_BURST, BURSTS_IN_FLIGHT>(
        din0, cube_words[0], height, width);
    read_rows_ahead<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, READ_PORTS, 1, PIXELS_PER_BURST, BURSTS_IN_FLIGHT>(
        din1, cube_words[1], height, width);
    pixel_buffer<PATCH_SIZE, PATCH_SIZE, WORDS_PER_PIXEL, FIRST_KEPT_WORD, BANDS_PER_WORD, DWT_BANDS_PER_CYCLE,
                 READ_PORTS, DWT_CYCLES_PER_PIXEL, zscore::FRAC, zscore::IN_MAX, zscore::A, zscore::B,
                 ZSCORE_ON_DSPS>(cube_words, zscores, height, width);

    // The wavelet transform's taps in LUTs (8 constants), the projection 226 -> 12 on DSPs.
    dwt_proj<PATCH_SIZE, PATCH_SIZE, MODEL_BANDS, DWT_TAPS, MID_CHANNELS, DWT_STEPS_PER_CYCLE,
             weights::dwt_spectral_h0_frac, PROJECTION_SUM_FRAC_BITS - weights::projection_bias_frac,
             PROJECTION_SUM_FRAC_BITS - ACTIVATION_FRAC_BITS, weights::dwt_spectral_h0, weights::dwt_spectral_h1,
             weights::projection_weight, weights::projection_bias, false, true>(zscores, projected, height, width);

    // The 3x3 conv, 12 -> 12.
    conv3x3_serial<PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, MID_CHANNELS, MID_CHANNELS, CONV_IN_PER_CYCLE,
                   CONV_SUM_FRAC_BITS - weights::conv_block_0_bias_frac, weights::conv_block_0_weight,
                   weights::conv_block_0_bias, true>(projected, conv_sums, height, width);

    // LeakyReLU, the 1x1 head 12 -> 3, LeakyReLU: a pixel a cycle.
    leaky_head<PATCH_SIZE, PATCH_SIZE, MID_CHANNELS, CLASSES, CONV_SUM_FRAC_BITS - ACTIVATION_FRAC_BITS,
               HEAD_SUM_FRAC_BITS - weights::seg_head_0_bias_frac, HEAD_SUM_FRAC_BITS - SCORE_FRAC_BITS,
               weights::seg_head_0_weight, weights::seg_head_0_bias, true>(conv_sums, scores, height, width);

    store_scores<PATCH_SIZE, PATCH_SIZE, 0>(scores, dout, height, width);
}
