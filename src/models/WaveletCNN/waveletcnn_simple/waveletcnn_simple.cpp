#include "waveletcnn_simple.hpp"

#include "../../../../weights/quantized/waveletcnn/waveletcnn_int8.hpp"
namespace weights = waveletcnn_int8;
namespace zs = zscore_int8;

// A layer's sums have its input's fraction bits plus its weights'; the bias is shifted up to
// them, the sums down to the next layer's. The wavelet transform's sums keep all of theirs
// (exact in int16: |z| <= 127 times the filter's 239 at most).
template<int IN_FRAC, int WEIGHT_FRAC, int BIAS_FRAC, int OUT_FRAC>
struct fracs {
    static constexpr int SUM = IN_FRAC + WEIGHT_FRAC;
    static constexpr int BIAS_SHIFT = SUM - BIAS_FRAC;
    static constexpr int SHIFT = SUM - OUT_FRAC;
};
constexpr int DWT_FRAC_BITS = ZSCORE_FRAC_BITS + weights::dwt_spectral_h0_frac;
static_assert(weights::dwt_spectral_h0_frac == weights::dwt_spectral_h1_frac, "both filters at the same fraction bits");
typedef fracs<DWT_FRAC_BITS, weights::projection_weight_frac, weights::projection_bias_frac, ACTIVATION_FRAC_BITS> FP;
typedef fracs<ACTIVATION_FRAC_BITS, weights::conv_block_0_weight_frac, weights::conv_block_0_bias_frac,
              ACTIVATION_FRAC_BITS> FC;
typedef fracs<ACTIVATION_FRAC_BITS, weights::seg_head_0_weight_frac, weights::seg_head_0_bias_frac, SCORE_FRAC_BITS> FH;
constexpr int P = PATCH_SIZE, MID = MID_CHANNELS;
typedef int16_t act_t;

// Each layer's work a step, the fewest multipliers that keep it within 36 cycles a pixel
// (~37 000 cycles a patch, ~100 ms an image at 250 MHz):
//              in -> out   UI   UO   cycles a pixel  multipliers
// dwt         110 -> 226   2 low and 2 high a step   29   (constant taps: shifts and adds)
// projection  226 ->  12  113    1        24             113
// conv 3x3     12 ->  12   12    3        36              36
// head         12 ->   3    1    1        36               1
// The read takes 15 cycles a pixel (15 words).
void waveletcnn_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)
{
    // depth is for co-simulation only: the testbench's 40 x 50 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=2000
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // A DATAFLOW loop: each layer a process, so every layer works on a different patch at
    // the same time. The 3x3 conv's input is a map (a ping-pong buffer), the rest streams.
    const int count = simple::patch_count<P>(height, width);
    patches: for (int p = 0; p < count; p++) {
        #pragma HLS DATAFLOW
        SIMPLE_MAP simple::word<act_t, MID> projected[P][P];
        #pragma HLS AGGREGATE variable=projected
        simple::stream<int8_t, MODEL_BANDS> zscores;
        simple::stream<int16_t, WAVELET_CHANNELS> wavelet;   // [x * 2^7, low, high]
        simple::stream<act_t, MID> activated;
        simple::stream<int32_t, CLASSES> scores;

        simple::read_zscore<P, MODEL_BANDS, WORDS_PER_PIXEL, zs::FIRST_BAND / BANDS_PER_WORD, zs::FRAC, zs::IN_MAX,
                            int32_t, zs::A, zs::B>(din0, p, height, width, zscores);
        simple::dwt<P * P, MODEL_BANDS, DWT_TAPS, 2, weights::dwt_spectral_h0_frac, weights::dwt_spectral_h0,
                    weights::dwt_spectral_h1>(zscores, wavelet);
        // The projection: 226 -> 12, no activation, saturated to int16.
        simple::conv<P, P, WAVELET_CHANNELS, MID, 1, 113, 1, simple::CLAMP16, 0, FP::BIAS_SHIFT, FP::SHIFT, false,
                     weights::projection_weight, weights::projection_bias>(wavelet, projected);
        // conv_block: 3x3, 12 -> 12, LeakyReLU, saturated to int16.
        simple::conv<P, P, MID, MID, 3, 12, 3, simple::LEAKY_CLAMP16, 0, FC::BIAS_SHIFT, FC::SHIFT, false,
                     weights::conv_block_0_weight, weights::conv_block_0_bias>(projected, activated);
        // seg_head: 1x1, 12 -> 3, LeakyReLU: the scores.
        simple::conv<P, P, MID, CLASSES, 1, 1, 1, simple::LEAKY, 0, FH::BIAS_SHIFT, FH::SHIFT, false,
                     weights::seg_head_0_weight, weights::seg_head_0_bias>(activated, scores);
        simple::write_scores<P, CLASSES>(scores, dout, p, height, width);
    }
}
