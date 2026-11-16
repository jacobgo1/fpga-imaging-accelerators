#include "spectralpointwise1d_simple.hpp"

// Generated data: the 12-bit z-score and the uint8 activations' fraction bits.
#include "../../1D-Justo-LiuNet/justoliunet_opt/justoliunet_opt_formats.hpp"
#include "../spectralpointwise1d_opt/spectralpointwise1d_opt_formats.hpp"
#include "../../../../weights/quantized/spectralpointwise1d/spectralpointwise1d_int8.hpp"
namespace weights = spectralpointwise1d_int8;
namespace jf = justoliunet_opt_formats;
namespace fmt = spectralpointwise1d_formats;

// A layer's sums have its input's fraction bits plus its weights'; the bias is shifted
// up to them, the sums to the output's (the scores: up to OUT_FRAC, exactly).
template<int IN_FRAC, int WEIGHT_FRAC, int BIAS_FRAC, int OUT_FRAC>
struct fracs {
    static constexpr int BIAS_SHIFT = IN_FRAC + WEIGHT_FRAC - BIAS_FRAC;
    static constexpr int SHIFT = IN_FRAC + WEIGHT_FRAC - OUT_FRAC;
};
typedef fracs<jf::IN_FRAC, weights::convsp1_weight_frac, weights::convsp1_bias_frac, fmt::ACT_FRAC1> F1;
typedef fracs<fmt::ACT_FRAC1, weights::convsp2_weight_frac, weights::convsp2_bias_frac, fmt::ACT_FRAC2> F2;
typedef fracs<fmt::ACT_FRAC2, weights::convsp3_weight_frac, weights::convsp3_bias_frac, jf::OUT_FRAC> F3;

constexpr int H1 = sizeof(weights::convsp1_bias), H2 = sizeof(weights::convsp2_bias);
typedef int16_t z_t;     // the 12-bit z-scores
typedef uint8_t act_t;   // the uint8 activations

// One read port gives a word (8 bands) a cycle: 15 cycles a pixel. Each layer keeps
// under that (cycles a pixel = IN/UI * OUT/UO):
//            in -> out   UI  UO  cycles  multipliers
// convsp1   110 -> 24    22   8    15     176
// convsp2    24 -> 12    24   1    12      24
// convsp3    12 ->  3     4   1     9       4
void spectralpointwise1d_simple(const cube_word_t* din0, score_word_t* dout, int height, int width)
{
    // depth is for co-simulation only: the testbench's 9 x 7 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=945 max_read_burst_length=256
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=63
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    // A DATAFLOW loop: each layer a process, the blocks between them ping-pong buffers.
    const int pixels = height * width;
    const int count = simple::block_count<BLOCK>(pixels);
    blocks: for (int b = 0; b < count; b++) {
        #pragma HLS DATAFLOW
        SIMPLE_MAP simple::word<z_t, MODEL_BANDS> zscores[1][BLOCK];
        SIMPLE_MAP simple::word<act_t, H1> h1[1][BLOCK];
        SIMPLE_MAP simple::word<act_t, H2> h2[1][BLOCK];
        SIMPLE_MAP simple::word<int32_t, CLASSES> scores[1][BLOCK];

        simple::read_block<BLOCK, MODEL_BANDS, WORDS_PER_PIXEL, jf::FIRST_BAND / BANDS_PER_WORD, jf::PREP_FRAC, jf::IN_MAX,
                           int64_t, jf::A, jf::B>(din0, b, pixels, zscores);
        simple::conv<1, BLOCK, MODEL_BANDS, H1, 1, 22, 8, true, jf::ACT_MAX, F1::BIAS_SHIFT, F1::SHIFT, false,
                     weights::convsp1_weight, weights::convsp1_bias>(zscores, h1);
        simple::conv<1, BLOCK, H1, H2, 1, 24, 1, true, jf::ACT_MAX, F2::BIAS_SHIFT, F2::SHIFT, false,
                     weights::convsp2_weight, weights::convsp2_bias>(h1, h2);
        simple::conv<1, BLOCK, H2, CLASSES, 1, 4, 1, false, 0, F3::BIAS_SHIFT, F3::SHIFT, false,
                     weights::convsp3_weight, weights::convsp3_bias>(h2, scores);
        simple::write_block<BLOCK, CLASSES>(scores, dout, b, pixels);
    }
}
