// The 1D-Justo-LiuNet chain of the simple kernels, shared by justoliunet_simple and
// justoliunet_bn_simple. The kernel's .cpp first includes its header, its weights as
// namespace weights and its activations' fraction bits as namespace fmt, then this. The
// layers are justoliunet_golden's (src/models/1D-Justo-LiuNet/justoliunet_golden/), the
// integer formats justoliunet_opt's.

#include "../justoliunet_opt/justoliunet_opt_formats.hpp"
namespace jf = justoliunet_opt_formats;
typedef int16_t z_t;     // the 12-bit z-scores
typedef uint8_t act_t;   // the uint8 activations

// A layer's sums have its input's fraction bits plus its weights'; the bias is shifted up
// to them, the sums to the output's (the scores: up to OUT_FRAC, exactly).
template<int IN_FRAC, int WEIGHT_FRAC, int BIAS_FRAC, int OUT_FRAC>
struct fracs {
    static constexpr int BIAS_SHIFT = IN_FRAC + WEIGHT_FRAC - BIAS_FRAC;
    static constexpr int SHIFT = IN_FRAC + WEIGHT_FRAC - OUT_FRAC;
};
typedef fracs<jf::IN_FRAC, weights::conv1_weight_frac, weights::conv1_bias_frac, fmt::ACT_FRAC1> F1;
typedef fracs<fmt::ACT_FRAC1, weights::conv2_weight_frac, weights::conv2_bias_frac, fmt::ACT_FRAC2> F2;
typedef fracs<fmt::ACT_FRAC2, weights::conv3_weight_frac, weights::conv3_bias_frac, fmt::ACT_FRAC3> F3;
typedef fracs<fmt::ACT_FRAC3, weights::conv4_weight_frac, weights::conv4_bias_frac, fmt::ACT_FRAC4> F4;
typedef fracs<fmt::ACT_FRAC4, weights::fc_weight_frac, weights::fc_bias_frac, jf::OUT_FRAC> FD;

// The spectrum's length through the blocks: 110 -> conv 105 -> pool 52 -> 47 -> 23 -> 18 -> 9
// -> 4 -> 2; the channels 1 -> 6 -> 12 -> 18 -> 24.
constexpr int L1 = MODEL_BANDS - KERNEL_SIZE + 1, P1 = L1 / 2;
constexpr int L2 = P1 - KERNEL_SIZE + 1, P2 = L2 / 2;
constexpr int L3 = P2 - KERNEL_SIZE + 1, P3 = L3 / 2;
constexpr int L4 = P3 - KERNEL_SIZE + 1, P4 = L4 / 2;

// Each layer's positions (UP), inputs (UI) and outputs (UO) a step, all 6 taps at once, the
// fewest multipliers that keep it within 48 cycles a pixel (UP x UI x UO x 6 multipliers;
// cycles a pixel = positions/UP * IN/UI * OUT/UO):
//            in -> out   positions  UP  UI  UO  cycles  multipliers
// conv1       1 ->  6      105       7   1   2    45       84
// conv2       6 -> 12       47       1   6  12    47      432
// conv3      12 -> 18       18       3   3   9    48      486
// conv4      18 -> 24        4       1   3  12    48      216
// dense      48 ->  3        1       1   2   1    36        4
// The read takes 15 cycles a pixel (15 words).
#define SIMPLE_CONV1D(L, IN, OUT, UP, UI, UO, F, layer, input, output) \
    simple::conv1d<BLOCK, L, IN, OUT, KERNEL_SIZE, UP, UI, UO, simple::RELU, jf::ACT_MAX, F::BIAS_SHIFT, F::SHIFT, \
                   weights::layer##_weight, weights::layer##_bias>(input, output)

static void justoliunet_simple_chain(const cube_word_t* din0, score_word_t* dout, int height, int width) {
    // A DATAFLOW loop: each layer a process, a pixel at a time through streams.
    const int pixels = height * width;
    const int count = simple::block_count<BLOCK>(pixels);
    blocks: for (int b = 0; b < count; b++) {
        #pragma HLS DATAFLOW
        simple::stream<z_t, MODEL_BANDS> zscores;   // 110 positions of 1 channel
        simple::stream<act_t, L1 * 6> conv1;
        simple::stream<act_t, P1 * 6> pool1;
        simple::stream<act_t, L2 * 12> conv2;
        simple::stream<act_t, P2 * 12> pool2;
        simple::stream<act_t, L3 * 18> conv3;
        simple::stream<act_t, P3 * 18> pool3;
        simple::stream<act_t, L4 * 24> conv4;
        simple::stream<act_t, P4 * 24> pool4;
        simple::stream<int32_t, CLASSES> scores;

        simple::read_block<BLOCK, MODEL_BANDS, WORDS_PER_PIXEL, jf::FIRST_BAND / BANDS_PER_WORD, jf::PREP_FRAC, jf::IN_MAX,
                           int64_t, jf::A, jf::B>(din0, b, pixels, zscores);
        SIMPLE_CONV1D(MODEL_BANDS, 1, 6, 7, 1, 2, F1, conv1, zscores, conv1);
        simple::maxpool1d<BLOCK, L1, 6>(conv1, pool1);
        SIMPLE_CONV1D(P1, 6, 12, 1, 6, 12, F2, conv2, pool1, conv2);
        simple::maxpool1d<BLOCK, L2, 12>(conv2, pool2);
        SIMPLE_CONV1D(P2, 12, 18, 3, 3, 9, F3, conv3, pool2, conv3);
        simple::maxpool1d<BLOCK, L3, 18>(conv3, pool3);
        SIMPLE_CONV1D(P3, 18, 24, 1, 3, 12, F4, conv4, pool3, conv4);
        simple::maxpool1d<BLOCK, L4, 24>(conv4, pool4);
        // Flatten channel by channel and the dense layer: the scores, at OUT_FRAC fraction bits.
        simple::dense<BLOCK, P4, 24, CLASSES, 2, 1, FD::BIAS_SHIFT, FD::SHIFT, weights::fc_weight, weights::fc_bias>(
            pool4, scores);
        simple::write_block<BLOCK, CLASSES>(scores, dout, b, pixels);
    }
}

#undef SIMPLE_CONV1D
