// The SpecPW-UNet chain of the simple kernels, shared by the three sizes. The size's .cpp
// first includes its header, its weights as namespace weights and its table as namespace
// sizes (a layer_size per conv), then this. The layers are specpw_unet_fixed_golden's
// (tb/sp_unet_small_opt/specpw_unet_fixed_golden.hpp), with the upsampling done by the
// conv after it and the concat by double4's first conv.

namespace zs = zscore_int8;
typedef int16_t act_t;
constexpr int ACT_MAX = 32767;

// The channel counts, from the weights (int8: a byte each).
constexpr int STEM1 = sizeof(weights::convsp1_bias), STEM2 = sizeof(weights::convsp2_bias);
constexpr int STEM3 = sizeof(weights::convsp3_bias);
constexpr int DOUBLE1 = sizeof(weights::double1_block_0_bias), DOUBLE2 = sizeof(weights::double2_block_0_bias);
constexpr int DOUBLE3 = sizeof(weights::double3_block_0_bias), DOUBLE4 = sizeof(weights::double4_block_0_bias);
constexpr int P = PATCH_SIZE, HALF = PATCH_SIZE / 2, QUARTER = PATCH_SIZE / 4;

// A layer's sums have its input's fraction bits plus its weights'; the bias is
// shifted up to them, the sums rounded down to the output's.
template<int IN_FRAC, int WEIGHT_FRAC, int BIAS_FRAC, int OUT_FRAC>
struct fracs {
    static constexpr int BIAS_SHIFT = IN_FRAC + WEIGHT_FRAC - BIAS_FRAC;
    static constexpr int SHIFT = IN_FRAC + WEIGHT_FRAC - OUT_FRAC;
};
#define LAYER_FRACS(layer, in_frac, out_frac) \
    fracs<in_frac, weights::layer##_weight_frac, weights::layer##_bias_frac, out_frac>
#define ACT ACTIVATION_FRAC_BITS

// One conv of the chain: H x H, IN -> OUT, K x K, its table entry, its activation, its input
// upsampled or not.
#define SIMPLE_CONV(H, IN, OUT, K, SIZE, ACTIVATION, UP, layer, in_frac, out_frac, input, output)     simple::conv<H, H, IN, OUT, K, sizes::SIZE::UI, sizes::SIZE::UO, simple::ACTIVATION, ACT_MAX,                  LAYER_FRACS(layer, in_frac, out_frac)::BIAS_SHIFT, LAYER_FRACS(layer, in_frac, out_frac)::SHIFT, UP,                  weights::layer##_weight, weights::layer##_bias>(input, output)

static void sp_unet_simple(const cube_word_t* din0, score_word_t* dout, int height, int width) {
    const int count = simple::patch_count<P>(height, width);
    patches: for (int p = 0; p < count; p++) {
        #pragma HLS DATAFLOW
        // A 3x3 conv's input is a map (a ping-pong buffer); everything else goes a pixel at a
        // time through a stream, so the 1x1 convs and the pools need no map.
        SIMPLE_MAP simple::word<act_t, STEM3> stem3[P][P];
        SIMPLE_MAP simple::word<act_t, DOUBLE1> double1_first[P][P], skip[P][P];
        SIMPLE_MAP simple::word<act_t, DOUBLE1> pooled1[HALF][HALF];
        SIMPLE_MAP simple::word<act_t, DOUBLE2> double2_first[HALF][HALF], pooled2[QUARTER][QUARTER], up1[HALF][HALF];
        SIMPLE_MAP simple::word<act_t, DOUBLE3> double3_first[HALF][HALF], double3[HALF][HALF], up2[P][P];
        SIMPLE_MAP simple::word<act_t, DOUBLE4> double4_first[P][P];
        // Each map one memory as wide as its pixel word, not one per channel.
        #pragma HLS AGGREGATE variable=stem3
        #pragma HLS AGGREGATE variable=double1_first
        #pragma HLS AGGREGATE variable=skip
        #pragma HLS AGGREGATE variable=pooled1
        #pragma HLS AGGREGATE variable=double2_first
        #pragma HLS AGGREGATE variable=pooled2
        #pragma HLS AGGREGATE variable=up1
        #pragma HLS AGGREGATE variable=double3_first
        #pragma HLS AGGREGATE variable=double3
        #pragma HLS AGGREGATE variable=up2
        #pragma HLS AGGREGATE variable=double4_first
        // The skip waits for double4 while 8 processes work on later patches: room for
        // that many patches, else the chain stalls on it.
        #pragma HLS STREAM variable=skip type=pipo depth=10
        #pragma HLS BIND_STORAGE variable=skip type=ram_2p impl=uram
        simple::stream<int8_t, MODEL_BANDS> zscores;
        simple::stream<act_t, STEM1> stem1;
        simple::stream<act_t, STEM2> stem2;
        simple::stream<act_t, DOUBLE1> double1, pool1_in;
        simple::stream<act_t, DOUBLE2> double2;
        simple::stream<act_t, DOUBLE4> double4;
        simple::stream<int32_t, CLASSES> scores;

        simple::read_zscore<P, MODEL_BANDS, WORDS_PER_PIXEL, zs::FIRST_BAND / BANDS_PER_WORD, zs::FRAC, zs::IN_MAX,
                            int32_t, zs::A, zs::B>(din0, p, height, width, zscores);

        // The spectral stem, 1x1 convs.
        SIMPLE_CONV(P, MODEL_BANDS, STEM1, 1, CONVSP1, RELU, false, convsp1, ZSCORE_FRAC_BITS, ACT, zscores, stem1);
        SIMPLE_CONV(P, STEM1, STEM2, 1, CONVSP2, RELU, false, convsp2, ACT, ACT, stem1, stem2);
        SIMPLE_CONV(P, STEM2, STEM3, 1, CONVSP3, RELU, false, convsp3, ACT, ACT, stem2, stem3);

        // The encoder: double1 at 32 x 32, its output also the skip; double2 at 16 x 16.
        SIMPLE_CONV(P, STEM3, DOUBLE1, 3, DOUBLE1_CONV1, RELU, false, double1_block_0, ACT, ACT, stem3, double1_first);
        SIMPLE_CONV(P, DOUBLE1, DOUBLE1, 3, DOUBLE1_CONV2, RELU, false, double1_block_3, ACT, ACT, double1_first, double1);
        simple::duplicate<P, P, DOUBLE1>(double1, skip, pool1_in);
        simple::maxpool2<P, P, DOUBLE1>(pool1_in, pooled1);
        SIMPLE_CONV(HALF, DOUBLE1, DOUBLE2, 3, DOUBLE2_CONV1, RELU, false, double2_block_0, ACT, ACT, pooled1, double2_first);
        SIMPLE_CONV(HALF, DOUBLE2, DOUBLE2, 3, DOUBLE2_CONV2, RELU, false, double2_block_3, ACT, ACT, double2_first, double2);
        simple::maxpool2<HALF, HALF, DOUBLE2>(double2, pooled2);

        // The decoder: up1 on pooled2 upsampled to 16 x 16, double3, up2 on it upsampled to 32 x 32.
        SIMPLE_CONV(HALF, DOUBLE2, DOUBLE2, 3, UP1_CONV, RELU, true, up1_1, ACT, ACT, pooled2, up1);
        SIMPLE_CONV(HALF, DOUBLE2, DOUBLE3, 3, DOUBLE3_CONV1, RELU, false, double3_block_0, ACT, ACT, up1, double3_first);
        SIMPLE_CONV(HALF, DOUBLE3, DOUBLE3, 3, DOUBLE3_CONV2, RELU, false, double3_block_3, ACT, ACT, double3_first, double3);
        SIMPLE_CONV(P, DOUBLE3, DOUBLE3, 3, UP2_CONV, RELU, true, up2_1, ACT, ACT, double3, up2);

        // double4 on torch.cat([up2, skip]): up2's channels first (the other way round
        // the accuracy drops to 30-40 %).
        simple::conv_concat<P, P, DOUBLE3, DOUBLE1, DOUBLE4, 3, sizes::DOUBLE4_CONV1::UI, sizes::DOUBLE4_CONV1::UO,
                            simple::RELU, ACT_MAX, LAYER_FRACS(double4_block_0, ACT, ACT)::BIAS_SHIFT,
                            LAYER_FRACS(double4_block_0, ACT, ACT)::SHIFT, false,
                            weights::double4_block_0_weight, weights::double4_block_0_bias>(up2, skip, double4_first);
        SIMPLE_CONV(P, DOUBLE4, DOUBLE4, 3, DOUBLE4_CONV2, RELU, false, double4_block_3, ACT, ACT, double4_first, double4);

        // The scores: a 1x1 conv, no activation.
        SIMPLE_CONV(P, DOUBLE4, CLASSES, 1, FINAL_CONV, NONE, false, final, ACT, SCORE_FRAC_BITS, double4, scores);
        simple::write_scores<P, CLASSES>(scores, dout, p, height, width);
    }
}

#undef SIMPLE_CONV
#undef ACT
#undef LAYER_FRACS
