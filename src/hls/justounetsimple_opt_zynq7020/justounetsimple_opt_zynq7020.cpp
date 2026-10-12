#include "justounetsimple_opt_zynq7020.hpp"
#include "../justounetsimple_opt/justounetsimple_prep.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/zynq7020/patch_buffer.hpp"
#include "../../optimized/zynq7020/conv3x3_wide.hpp"
#include "../../optimized/shared/conv3x3_serial.hpp"
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
static_assert(prep::IN_FRAC == IN_FRAC && prep::IN_MAX == 127,
              "justounetsimple_prep.hpp is out of date: run python tools/justounetsimple_model.py");

constexpr int WORDS = RAW_BANDS / PORT_BANDS;           // 30 words per pixel in the cube
constexpr int FIRST = prep::FIRST_BAND / PORT_BANDS;    // words 0-1 (raw bands 0-7) are dropped
constexpr int KEPT = WORDS - FIRST;                      // words 2-29: kept bands 0-111
static_assert(RAW_BANDS % PORT_BANDS == 0 && P1 % PORT_BANDS == 0, "a chunk is whole words");
static_assert(prep::FIRST_BAND % PORT_BANDS == 0 && KEPT * PORT_BANDS == IN_PAD &&
              sizeof(prep::A) / sizeof(prep::A[0]) == KEPT * PORT_BANDS,
              "the kept bands must be whole words, to the end of each pixel, and conv1's input");

static_assert(OUT_CH + 1 == 4, "a pixel of dout is 3 scores and the class");

// Fraction bits: each layer's sums carry its input's plus its weights'.
constexpr int SUM1 = IN_FRAC + q::conv1_weight_frac;
constexpr int SUM2 = ACT_FRAC + q::conv2_weight_frac;
constexpr int SUM3 = ACT_FRAC + q::conv3_weight_frac;
constexpr int SUM4 = ACT_FRAC + q::conv4_weight_frac;
static_assert(SUM1 >= q::conv1_bias_frac && SUM2 >= q::conv2_bias_frac &&
              SUM3 >= q::conv3_bias_frac && SUM4 >= q::conv4_bias_frac,
              "a bias has more fraction bits than its layer's sums");
static_assert(SUM1 > ACT_FRAC && SUM2 > ACT_FRAC && SUM3 > ACT_FRAC &&
              SUM4 >= OUT_FRAC, "a requantization would have to add bits");
// int32 never overflows, as in justounetsimple_opt: the testbench checks
// every sum it sees.

void justounetsimple_opt_zynq7020(const jopt_raw_t *din0, const jopt_raw_t *din1,
                                  hls::burst_maxi<jopt_out_t> dout, int height, int width)
{
    // Three AXI masters, each on its own 64-bit HP port to DDR
    // (boards/zynq7020/system.tcl): two read the cube in parallel (point both
    // at it) through HP0-1, one writes the scores through HP2.
    // depth is only for co-simulation: the testbench's 40 x 50 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem2 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<jopt_raw_t> s_raw[PORTS];
    hls::stream<vec_t<act8_t, P1> > s_in("s_in");
    hls::stream<vec_t<acc_t, BASE_CH> > s_conv1("s_conv1");
    hls::stream<vec_t<act_t, BASE_CH> > s_a("s_a");
    hls::stream<vec_t<acc_t, 2 * BASE_CH> > s_conv2("s_conv2");
    hls::stream<vec_t<act_t, 2 * BASE_CH> > s_b("s_b");
    hls::stream<vec_t<acc_t, BASE_CH> > s_conv3("s_conv3");
    hls::stream<vec_t<act_t, BASE_CH> > s_c("s_c");
    hls::stream<vec_t<acc_t, OUT_CH> > s_conv4("s_conv4");
    // The readers pause between rows (a new burst); let them run ahead.
    #pragma HLS STREAM variable=s_raw depth=64
    #pragma HLS STREAM variable=s_in depth=64
    // conv1 hands out a patch's 16 x 16 pooled pixels during its last chunk
    // only (1024 cycles), conv2 takes one per 36 cycles. Room for all of them,
    // so conv1 never waits for conv2.
    #pragma HLS STREAM variable=s_a depth=256
    // A patch row of scores, so conv4 keeps going while store starts a burst.
    #pragma HLS STREAM variable=s_conv4 depth=32

    // Every process walks the same grid of H x W patches, one after another.
    read_rows<H, W, WORDS, PORTS, 0>(din0, s_raw[0], height, width);
    read_rows<H, W, WORDS, PORTS, 1>(din1, s_raw[1], height, width);
    patch_buffer<H, W, WORDS, FIRST, PORT_BANDS, P1, PORTS, prep::FRAC, prep::IN_MAX, prep::A, prep::B, false>(
        s_raw, s_in, height, width);
    // conv1: 110 -> 6 at 32 x 32, 8 bands per cycle (112 with the zero ones),
    // 6 x 8 x 9 = 432 multiply-adds per cycle, two per DSP: 216 DSPs.
    conv3x3_wide<H, W, H, W, IN_CH, BASE_CH, P1, SUM1 - q::conv1_bias_frac,
                 q::conv1_weight, q::conv1_bias>(s_in, s_conv1, height, width);
    relu_pool<H, W, H, W, SUM1 - ACT_FRAC>(s_conv1, s_a, height, width);         // -> 16 x 16
    // conv2: 6 -> 12 at 16 x 16, 36 cycles per pixel.
    conv3x3_serial<H, W, H / 2, W / 2, BASE_CH, 2 * BASE_CH, P2, SUM2 - q::conv2_bias_frac,
                   q::conv2_weight, q::conv2_bias, false>(s_a, s_conv2, height, width);
    relu_pool_up<H, W, H / 2, W / 2, SUM2 - ACT_FRAC>(s_conv2, s_b, height, width);    // 16 x 16
    // conv3: 12 -> 6 at 16 x 16, 36 cycles per pixel.
    conv3x3_serial<H, W, H / 2, W / 2, 2 * BASE_CH, BASE_CH, P2, SUM3 - q::conv3_bias_frac,
                   q::conv3_weight, q::conv3_bias, false>(s_b, s_conv3, height, width);
    relu_up<H, W, H / 2, W / 2, SUM3 - ACT_FRAC>(s_conv3, s_c, height, width);         // -> 32 x 32
    // conv4: 6 -> 3 at 32 x 32, no activation, 9 cycles per pixel.
    conv3x3_serial<H, W, H, W, BASE_CH, OUT_CH, P2, SUM4 - q::conv4_bias_frac,
                   q::conv4_weight, q::conv4_bias, false>(s_c, s_conv4, height, width);
    store_scores<H, W, SUM4 - OUT_FRAC>(s_conv4, dout, height, width);
}
