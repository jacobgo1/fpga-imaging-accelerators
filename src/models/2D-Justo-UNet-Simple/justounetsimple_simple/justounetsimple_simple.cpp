#include "justounetsimple_simple.hpp"

#include "../../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"
namespace weights = justounetsimple_int8;
namespace zs = zscore_int8;

// A layer's sums have the fraction bits of its input plus those of its weights.
// The bias is shifted up to them, the sums down to the next layer's activations.
template<int IN_FRAC, int WEIGHT_FRAC, int BIAS_FRAC, int OUT_FRAC>
struct fracs {
    static constexpr int BIAS_SHIFT = IN_FRAC + WEIGHT_FRAC - BIAS_FRAC;
    static constexpr int SHIFT = IN_FRAC + WEIGHT_FRAC - OUT_FRAC;
};
typedef fracs<ZSCORE_FRAC_BITS, weights::conv1_weight_frac, weights::conv1_bias_frac, ACTIVATION_FRAC_BITS> F1;
typedef fracs<ACTIVATION_FRAC_BITS, weights::conv2_weight_frac, weights::conv2_bias_frac, ACTIVATION_FRAC_BITS> F2;
typedef fracs<ACTIVATION_FRAC_BITS, weights::conv3_weight_frac, weights::conv3_bias_frac, ACTIVATION_FRAC_BITS> F3;
typedef fracs<ACTIVATION_FRAC_BITS, weights::conv4_weight_frac, weights::conv4_bias_frac, SCORE_FRAC_BITS> F4;
constexpr int ACT_MAX = 32767;   // int16 activations

// Each layer's inputs (UI) and outputs (UO) a cycle, so none takes more than ~20 000
// cycles a patch (cycles = H * W * 9 * IN/UI * OUT/UO):
//            in -> out   size    UI  UO   cycles   multipliers
// conv1     110 ->  6   32 x 32  55   6   18 432   330
// conv2       6 -> 12   16 x 16   6   2   13 824    12
// conv3      12 ->  6   16 x 16  12   1   13 824    12
// conv4       6 ->  3   32 x 32   6   3    9 216    18
// The read takes ~15 400 cycles a patch (32 x 32 pixels x 15 words) plus a DDR wait a row.
constexpr int C = BASE_CHANNELS;
typedef int16_t act_t;

void justounetsimple_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)
{
    // depth is for co-simulation only: the testbench's 40 x 50 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=2000
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // A DATAFLOW loop: each layer a process, the maps between them ping-pong buffers,
    // so every layer works on a different patch at the same time.
    const int count = simple::patch_count<PATCH_SIZE>(height, width);
    patches: for (int p = 0; p < count; p++) {
        #pragma HLS DATAFLOW
        SIMPLE_MAP simple::word<int8_t, MODEL_BANDS> zscores[PATCH_SIZE][PATCH_SIZE];
        SIMPLE_MAP simple::word<act_t, C> relu1[PATCH_SIZE][PATCH_SIZE], pooled1[HALF_PATCH][HALF_PATCH];
        SIMPLE_MAP simple::word<act_t, 2 * C> relu2[HALF_PATCH][HALF_PATCH], pooled2[HALF_PATCH / 2][HALF_PATCH / 2];
        SIMPLE_MAP simple::word<act_t, C> relu3[HALF_PATCH][HALF_PATCH];
        SIMPLE_MAP simple::word<int32_t, CLASSES> scores[PATCH_SIZE][PATCH_SIZE];

        simple::read_zscore<PATCH_SIZE, MODEL_BANDS, WORDS_PER_PIXEL, zs::FIRST_BAND / BANDS_PER_WORD, zs::FRAC, zs::IN_MAX,
                            int32_t, zs::A, zs::B>(din0, p, height, width, zscores);
        simple::conv<PATCH_SIZE, PATCH_SIZE, MODEL_BANDS, C, 3, 55, 6, true, ACT_MAX, F1::BIAS_SHIFT, F1::SHIFT, false,
                     weights::conv1_weight, weights::conv1_bias>(zscores, relu1);
        simple::maxpool2<PATCH_SIZE, PATCH_SIZE, C>(relu1, pooled1);
        simple::conv<HALF_PATCH, HALF_PATCH, C, 2 * C, 3, 6, 2, true, ACT_MAX, F2::BIAS_SHIFT, F2::SHIFT, false,
                     weights::conv2_weight, weights::conv2_bias>(pooled1, relu2);
        simple::maxpool2<HALF_PATCH, HALF_PATCH, 2 * C>(relu2, pooled2);
        // conv3 on pooled2 upsampled to 16 x 16, conv4 on relu3 upsampled to 32 x 32.
        simple::conv<HALF_PATCH, HALF_PATCH, 2 * C, C, 3, 12, 1, true, ACT_MAX, F3::BIAS_SHIFT, F3::SHIFT, true,
                     weights::conv3_weight, weights::conv3_bias>(pooled2, relu3);
        simple::conv<PATCH_SIZE, PATCH_SIZE, C, CLASSES, 3, 6, 3, false, ACT_MAX, F4::BIAS_SHIFT, F4::SHIFT, true,
                     weights::conv4_weight, weights::conv4_bias>(relu3, scores);
        simple::write_scores<PATCH_SIZE, CLASSES>(scores, dout, p, height, width);
    }
}
