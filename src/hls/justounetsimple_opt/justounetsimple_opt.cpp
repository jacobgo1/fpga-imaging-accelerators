#include "justounetsimple_opt.hpp"
#include "justounetsimple_prep.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/zcu104/patch_buffer.hpp"
#include "../../optimized/zcu104/conv3x3_wide.hpp"
#include "../../optimized/shared/conv3x3_narrow.hpp"
#include "../../optimized/shared/relu_pool.hpp"
#include "../../optimized/shared/relu_pool_up.hpp"
#include "../../optimized/shared/relu_up.hpp"
#include "../../optimized/shared/store_scores.hpp"
#include <type_traits>

// The quantized weights, as the int8 integers q they are (value q * 2^-frac):
// this kernel multiplies integers. The scales become shifts, below.
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace q = justounetsimple_int8;
namespace prep = justounetsimple_prep;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::conv1_weight)>::type>::type, wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(prep::IN_FRAC == JOPT_IN_FRAC && prep::IN_MAX == 127,
              "justounetsimple_prep.hpp is out of date: run python tools/justounetsimple_model.py");

constexpr int H = JOPT_H, W = JOPT_W, IN = JOPT_IN_CH, B = JOPT_BASE_CH, OUT = JOPT_OUT_CH;
constexpr int P1 = JOPT_P1, INP = JOPT_IN_PAD, PB = JOPT_PORT_BANDS, PORTS = JOPT_PORTS;
constexpr int WORDS = JOPT_RAW_BANDS / PB;      // 15 words per pixel in the cube
constexpr int FIRST = prep::FIRST_BAND / PB;    // word 0 (raw bands 0-7) is dropped
constexpr int KEPT = WORDS - FIRST;             // words 1-14: kept bands 0-111
static_assert(JOPT_RAW_BANDS % PB == 0 && P1 % PB == 0, "a chunk is whole words");
static_assert(prep::FIRST_BAND % PB == 0 && KEPT * PB >= IN && KEPT * PB <= INP &&
              sizeof(prep::A) / sizeof(prep::A[0]) == KEPT * PB,
              "the kept bands must be whole words, to the end of each pixel");

// Fraction bits: each layer's sums carry its input's plus its weights'.
constexpr int SUM1 = JOPT_IN_FRAC + q::conv1_weight_frac;
constexpr int SUM2 = JOPT_ACT_FRAC + q::conv2_weight_frac;
constexpr int SUM3 = JOPT_ACT_FRAC + q::conv3_weight_frac;
constexpr int SUM4 = JOPT_ACT_FRAC + q::conv4_weight_frac;
static_assert(SUM1 >= q::conv1_bias_frac && SUM2 >= q::conv2_bias_frac &&
              SUM3 >= q::conv3_bias_frac && SUM4 >= q::conv4_bias_frac,
              "a bias has more fraction bits than its layer's sums");
static_assert(SUM1 > JOPT_ACT_FRAC && SUM2 > JOPT_ACT_FRAC && SUM3 > JOPT_ACT_FRAC &&
              SUM4 >= JOPT_OUT_FRAC, "a requantization would have to add bits");
// int32 never overflows: the largest |sum| a layer can make is the largest
// input times the largest sum of |q| over one output channel: 127 * ... = 1.4e6
// for conv1 (int8 input), 32767 * ... <= 7.7e7 for the others. The testbench
// checks every sum it sees.

void justounetsimple_opt(const jopt_raw_t *din0, const jopt_raw_t *din1,
                         const jopt_raw_t *din2, const jopt_raw_t *din3,
                         int32_t *dout, int height, int width)
{
    // Five AXI masters, each on its own port to DDR (boards/zcu104/system.tcl):
    // four read the cube in parallel (point all of them at it) through HP0-3,
    // one writes the scores through HPC0.
    // depth is only for co-simulation: the testbench's 40 x 50 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem4 offset=slave depth=6000 max_write_burst_length=128
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<jopt_raw_t> s_raw[PORTS];
    hls::stream<vec_t<act8_t, P1> > s_in("s_in");
    hls::stream<vec_t<acc_t, B> > s_conv1("s_conv1");
    hls::stream<vec_t<act_t, B> > s_a("s_a");
    hls::stream<vec_t<acc_t, 2 * B> > s_conv2("s_conv2");
    hls::stream<vec_t<act_t, 2 * B> > s_b("s_b");
    hls::stream<vec_t<acc_t, B> > s_conv3("s_conv3");
    hls::stream<vec_t<act_t, B> > s_c("s_c");
    hls::stream<vec_t<acc_t, OUT> > s_conv4("s_conv4");
    // The readers pause between rows (a new burst); let them run ahead.
    #pragma HLS STREAM variable=s_raw depth=64
    #pragma HLS STREAM variable=s_in depth=64
    // conv1 hands out a patch's 16 x 16 pooled pixels during its last chunk
    // only (1024 cycles), conv2 takes one per 12 cycles. Room for all of them,
    // so conv1 never waits for conv2.
    #pragma HLS STREAM variable=s_a depth=256
    // A patch row of scores, so conv4 keeps going while store starts a burst.
    #pragma HLS STREAM variable=s_conv4 depth=32

    // Every process walks the same grid of H x W patches, one after another.
    read_rows<H, W, WORDS, PORTS, 0>(din0, s_raw[0], height, width);
    read_rows<H, W, WORDS, PORTS, 1>(din1, s_raw[1], height, width);
    read_rows<H, W, WORDS, PORTS, 2>(din2, s_raw[2], height, width);
    read_rows<H, W, WORDS, PORTS, 3>(din3, s_raw[3], height, width);
    patch_buffer<H, W, WORDS, FIRST, PB, P1, PORTS, prep::FRAC, prep::IN_MAX, prep::A, prep::B>(
        s_raw, s_in, height, width);
    // conv1: 110 -> 6 at 32 x 32, 32 bands per cycle (128 with the zero ones),
    // 6 x 32 x 9 = 1728 multiply-adds per cycle, two per DSP: 864 DSPs.
    conv3x3_wide<H, W, H, W, IN, B, P1, SUM1 - q::conv1_bias_frac, q::conv1_weight, q::conv1_bias>(
        s_in, s_conv1, height, width);
    relu_pool<H, W, H, W, SUM1 - JOPT_ACT_FRAC>(s_conv1, s_a, height, width);              // -> 16 x 16
    // conv2: 6 -> 12 at 16 x 16, 12 cycles per pixel.
    conv3x3_narrow<H, W, H / 2, W / 2, B, 2 * B, SUM2 - q::conv2_bias_frac, q::conv2_weight, q::conv2_bias>(
        s_a, s_conv2, height, width);
    relu_pool_up<H, W, H / 2, W / 2, SUM2 - JOPT_ACT_FRAC>(s_conv2, s_b, height, width);   // 16 x 16
    // conv3: 12 -> 6 at 16 x 16, 6 cycles per pixel.
    conv3x3_narrow<H, W, H / 2, W / 2, 2 * B, B, SUM3 - q::conv3_bias_frac, q::conv3_weight, q::conv3_bias>(
        s_b, s_conv3, height, width);
    relu_up<H, W, H / 2, W / 2, SUM3 - JOPT_ACT_FRAC>(s_conv3, s_c, height, width);        // -> 32 x 32
    // conv4: 6 -> 3 at 32 x 32, no activation, 3 cycles per pixel.
    conv3x3_narrow<H, W, H, W, B, OUT, SUM4 - q::conv4_bias_frac, q::conv4_weight, q::conv4_bias>(
        s_c, s_conv4, height, width);
    store_scores<H, W, SUM4 - JOPT_OUT_FRAC>(s_conv4, dout, height, width);
}
