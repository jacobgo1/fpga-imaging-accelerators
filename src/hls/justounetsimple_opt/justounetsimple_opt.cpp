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
#include "../../optimized/shared/timeline.hpp"
#include <type_traits>

// The quantized weights, as the int8 integers q they are (value q * 2^-frac):
// this kernel multiplies integers. The scales become shifts, below.
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace q = justounetsimple_int8;
namespace prep = justounetsimple_prep;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                               decltype(q::conv1_weight)>::type>::type,
                           wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(prep::IN_FRAC == IN_FRAC && prep::IN_MAX == 127,
              "justounetsimple_prep.hpp is out of date: run python tools/justounetsimple_model.py");

constexpr int WORDS = RAW_BANDS / PORT_BANDS;        // 15 words per pixel in the cube
constexpr int FIRST = prep::FIRST_BAND / PORT_BANDS; // word 0 (raw bands 0-7) is dropped
constexpr int KEPT = WORDS - FIRST;                  // words 1-14: kept bands 0-111
static_assert(RAW_BANDS % PORT_BANDS == 0 && P1 % PORT_BANDS == 0, "a chunk is whole words");
static_assert(prep::FIRST_BAND % PORT_BANDS == 0 &&
                  KEPT * PORT_BANDS >= IN_CH && KEPT * PORT_BANDS <= IN_PAD &&
                  sizeof(prep::A) / sizeof(prep::A[0]) == KEPT * PORT_BANDS,
              "the kept bands must be whole words, to the end of each pixel");

static_assert(OUT_CH + 1 == 4, "a pixel of dout is 3 scores and the class");
static_assert(timeline_layout<TRACE_LANES, TRACE_PATCHES>::HEAD == TRACE_HEAD &&
                  timeline_layout<TRACE_LANES, TRACE_PATCHES>::WORDS == TRACE_WORDS,
              "the trace layout in the header must be timeline.hpp's");

// Fraction bits: each layer's sums carry its input's plus its weights'.
constexpr int SUM1 = IN_FRAC + q::conv1_weight_frac;
constexpr int SUM2 = ACT_FRAC + q::conv2_weight_frac;
constexpr int SUM3 = ACT_FRAC + q::conv3_weight_frac;
constexpr int SUM4 = ACT_FRAC + q::conv4_weight_frac;
static_assert(SUM1 >= q::conv1_bias_frac && SUM2 >= q::conv2_bias_frac &&
                  SUM3 >= q::conv3_bias_frac && SUM4 >= q::conv4_bias_frac,
              "a bias has more fraction bits than its layer's sums");
static_assert(SUM1 > ACT_FRAC && SUM2 > ACT_FRAC && SUM3 > ACT_FRAC &&
                  SUM4 >= OUT_FRAC,
              "a requantization would have to add bits");
// int32 never overflows: the largest |sum| a layer can make is the largest
// input times the largest sum of |q| over one output channel: 127 * ... = 1.4e6
// for conv1 (int8 input), 32767 * ... <= 7.7e7 for the others. The testbench
// checks every sum it sees.

void justounetsimple_opt(hls::burst_maxi<jopt_raw_t> din0, hls::burst_maxi<jopt_raw_t> din1,
                         hls::burst_maxi<jopt_raw_t> din2, hls::burst_maxi<jopt_raw_t> din3,
                         hls::burst_maxi<jopt_out_t> dout, int height, int width, uint32_t& cycles,
                         uint32_t *trace)
{
// Six AXI masters, each on its own port to DDR (boards/zcu104/system.tcl):
// four read the cube in parallel (point all of them at it) through HP0-3,
// one writes the scores through HPC0, one the timeline through HPC1.
// depth is only for co-simulation: the testbench's 40 x 50 image.
#pragma HLS INTERFACE mode = m_axi port = din0 bundle = gmem0 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = din1 bundle = gmem1 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = din2 bundle = gmem2 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = din3 bundle = gmem3 offset = slave depth = 30000 max_read_burst_length = 256 num_read_outstanding = 4
#pragma HLS INTERFACE mode = m_axi port = dout bundle = gmem4 offset = slave depth = 2000 max_write_burst_length = 32 num_write_outstanding = 16
#pragma HLS INTERFACE mode = m_axi port = trace bundle = gmem5 offset = slave depth = 24592 max_write_burst_length = 256
#pragma HLS INTERFACE mode = s_axilite port = din0 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = din1 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = din2 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = din3 bundle = control
#pragma HLS INTERFACE mode = s_axilite port = dout bundle = control
#pragma HLS INTERFACE mode = s_axilite port = height bundle = control
#pragma HLS INTERFACE mode = s_axilite port = width bundle = control
#pragma HLS INTERFACE mode = s_axilite port = cycles bundle = control
#pragma HLS INTERFACE mode = s_axilite port = trace bundle = control
#pragma HLS INTERFACE mode = s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    hls::stream<jopt_raw_t> s_raw[PORTS];
    hls::stream<vec_t<act8_t, P1>> s_in("s_in");
    hls::stream<vec_t<acc_t, BASE_CH>> s_conv1("s_conv1");
    hls::stream<vec_t<act_t, BASE_CH>> s_a("s_a");
    hls::stream<vec_t<acc_t, 2 * BASE_CH>> s_conv2("s_conv2");
    hls::stream<vec_t<act_t, 2 * BASE_CH>> s_b("s_b");
    hls::stream<vec_t<acc_t, BASE_CH>> s_conv3("s_conv3");
    hls::stream<vec_t<act_t, BASE_CH>> s_c("s_c");
    hls::stream<vec_t<acc_t, OUT_CH>> s_conv4("s_conv4");
    hls::stream<bool> s_finished("s_finished");
    // The timeline: each producer writes into o_*, a tap passes the words on to
    // the s_* above (with the depths below) and marks each patch on its lane.
    hls::stream<jopt_raw_t> o_raw[PORTS];
    hls::stream<vec_t<act8_t, P1>> o_in("o_in");
    hls::stream<vec_t<acc_t, BASE_CH>> o_conv1("o_conv1");
    hls::stream<vec_t<act_t, BASE_CH>> o_a("o_a");
    hls::stream<vec_t<acc_t, 2 * BASE_CH>> o_conv2("o_conv2");
    hls::stream<vec_t<act_t, 2 * BASE_CH>> o_b("o_b");
    hls::stream<vec_t<acc_t, BASE_CH>> o_conv3("o_conv3");
    hls::stream<vec_t<act_t, BASE_CH>> o_c("o_c");
    hls::stream<vec_t<acc_t, OUT_CH>> o_conv4("o_conv4");
    hls::stream<bool> s_ev[TRACE_LANES];
#pragma HLS STREAM variable = s_ev depth = 4
// Let the readers run ahead of patch_buffer, which takes a word from all four at once.
#pragma HLS STREAM variable = s_raw depth = 64
#pragma HLS STREAM variable = s_in depth = 64
// conv1 hands out a patch's 16 x 16 pooled pixels during its last chunk
// only (1024 cycles), conv2 takes one per 12 cycles. Room for all of them,
// so conv1 never waits for conv2.
#pragma HLS STREAM variable = s_a depth = 256
// relu_pool_up gives a row pair (32 pixels) at once and conv3 takes one per 6
// cycles; relu_up gives two rows (64 pixels) at once and conv4 takes one per 3.
// With the default depth of 2 they cannot take their next input while they
// give, conv2 and conv3 wait for them, and the chain behind conv1 took ~6 600
// cycles a patch instead of conv1's 4 129. Room for all of it:
#pragma HLS STREAM variable = s_b depth = 32
#pragma HLS STREAM variable = s_c depth = 64
// A patch row of scores, so conv4 keeps going while store starts a burst.
#pragma HLS STREAM variable = s_conv4 depth = 32

    // Every process walks the same grid of H x W patches, one after another.
    // Each asks for its next row one row ahead: at most 3 bursts outstanding,
    // within num_read_outstanding = 4.
    read_rows_ahead<H, W, WORDS, PORTS, 0>(din0, o_raw[0], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 1>(din1, o_raw[1], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 2>(din2, o_raw[2], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 3>(din3, o_raw[3], height, width);
    tap<H, W, H / PORTS * W * WORDS>(o_raw[0], s_raw[0], s_ev[0], height, width);
    tap<H, W, H / PORTS * W * WORDS>(o_raw[1], s_raw[1], s_ev[1], height, width);
    tap<H, W, H / PORTS * W * WORDS>(o_raw[2], s_raw[2], s_ev[2], height, width);
    tap<H, W, H / PORTS * W * WORDS>(o_raw[3], s_raw[3], s_ev[3], height, width);
    patch_buffer<H, W, WORDS, FIRST, PORT_BANDS, P1, PORTS, prep::FRAC, prep::IN_MAX, prep::A, prep::B, true>(
        s_raw, o_in, height, width);
    tap<H, W, H * W * (IN_PAD / P1)>(o_in, s_in, s_ev[4], height, width);
    // conv1: 110 -> 6 at 32 x 32, 32 bands per cycle (128 with the zero ones),
    // 6 x 32 x 9 = 1728 multiply-adds per cycle, two per DSP: 864 DSPs.
    conv3x3_wide<H, W, H, W, IN_CH, BASE_CH, P1, SUM1 - q::conv1_bias_frac,
                 q::conv1_weight, q::conv1_bias>(s_in, o_conv1, height, width);
    tap<H, W, H * W>(o_conv1, s_conv1, s_ev[5], height, width);
    relu_pool<H, W, H, W, SUM1 - ACT_FRAC>(s_conv1, o_a, height, width); // -> 16 x 16
    tap<H, W, H / 2 * W / 2>(o_a, s_a, s_ev[6], height, width);
    // conv2: 6 -> 12 at 16 x 16, 12 cycles per pixel.
    conv3x3_narrow<H, W, H / 2, W / 2, BASE_CH, 2 * BASE_CH, SUM2 - q::conv2_bias_frac,
                   q::conv2_weight, q::conv2_bias, true>(s_a, o_conv2, height, width);
    tap<H, W, H / 2 * W / 2>(o_conv2, s_conv2, s_ev[7], height, width);
    relu_pool_up<H, W, H / 2, W / 2, SUM2 - ACT_FRAC>(s_conv2, o_b, height, width); // 16 x 16
    tap<H, W, H / 2 * W / 2>(o_b, s_b, s_ev[8], height, width);
    // conv3: 12 -> 6 at 16 x 16, 6 cycles per pixel.
    conv3x3_narrow<H, W, H / 2, W / 2, 2 * BASE_CH, BASE_CH, SUM3 - q::conv3_bias_frac,
                   q::conv3_weight, q::conv3_bias, true>(s_b, o_conv3, height, width);
    tap<H, W, H / 2 * W / 2>(o_conv3, s_conv3, s_ev[9], height, width);
    relu_up<H, W, H / 2, W / 2, SUM3 - ACT_FRAC>(s_conv3, o_c, height, width); // -> 32 x 32
    tap<H, W, H * W>(o_c, s_c, s_ev[10], height, width);
    // conv4: 6 -> 3 at 32 x 32, no activation, 3 cycles per pixel.
    conv3x3_narrow<H, W, H, W, BASE_CH, OUT_CH, SUM4 - q::conv4_bias_frac,
                   q::conv4_weight, q::conv4_bias, true>(s_c, o_conv4, height, width);
    tap<H, W, H * W>(o_conv4, s_conv4, s_ev[11], height, width);
    store_scores<H, W, SUM4 - OUT_FRAC>(s_conv4, dout, height, width, s_finished);
    // Last: C simulation runs the processes in this order (timeline.hpp).
    timeline<TRACE_LANES, TRACE_PATCHES>(s_ev, s_finished, trace, cycles);
}
