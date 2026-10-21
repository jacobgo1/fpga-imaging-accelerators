#include "waveletcnn_opt.hpp"
#include "waveletcnn_prep.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/pixel_buffer.hpp"
#include "../../optimized/shared/dwt_proj.hpp"
#include "../../optimized/shared/conv3x3_narrow.hpp"
#include "../../optimized/shared/leaky_head.hpp"
#include "../../optimized/shared/store_scores.hpp"
#include "../../optimized/shared/count_cycles.hpp"
#include <type_traits>

// The quantized weights, as the int8 integers q they are (value q * 2^-frac):
// this kernel multiplies integers. The scales become shifts, below.
#include "../../../weights/quantized/waveletcnn/waveletcnn_int8.hpp"

namespace q = waveletcnn_int8;
namespace prep = waveletcnn_prep;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::projection_weight)>::type>::type, wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(prep::IN_FRAC == IN_FRAC && prep::IN_MAX == 127,
              "waveletcnn_prep.hpp is out of date: run python tools/waveletcnn_model.py");
static_assert(q::dwt_spectral_h0_frac == q::dwt_spectral_h1_frac,
              "dwt_spectral.h0 and h1 must share the same fraction bit count");

constexpr int WORDS = RAW_BANDS / PORT_BANDS;           // 15 words per pixel in the cube
constexpr int FIRST = prep::FIRST_BAND / PORT_BANDS;    // word 0 (raw bands 0-7) is dropped
constexpr int KEPT = WORDS - FIRST;                      // words 1-14: kept bands 0-111
static_assert(RAW_BANDS % PORT_BANDS == 0 && P1 % PORT_BANDS == 0, "a chunk is whole words");
static_assert(prep::FIRST_BAND % PORT_BANDS == 0 &&
              sizeof(prep::A) / sizeof(prep::A[0]) == KEPT * PORT_BANDS,
              "the kept bands must be whole words, to the end of each pixel");
static_assert(OUT_CH + 1 == 4, "a pixel of dout is 3 scores and the class");

// Fraction bits:
// - dwt_spectral products: IN_FRAC + h0_frac (exact in int16, no shift or clipping)
// - projection sums:       DWT_FRAC + projection_weight_frac
// - conv_block sums:       ACT_FRAC + conv_block_0_weight_frac
// - seg_head sums:         ACT_FRAC + seg_head_0_weight_frac
constexpr int DWT_FRAC = IN_FRAC + q::dwt_spectral_h0_frac;
constexpr int SUM_PROJ = DWT_FRAC + q::projection_weight_frac;
constexpr int SUM_CONV = ACT_FRAC + q::conv_block_0_weight_frac;
constexpr int SUM_HEAD = ACT_FRAC + q::seg_head_0_weight_frac;
static_assert(SUM_PROJ >= q::projection_bias_frac &&
              SUM_CONV >= q::conv_block_0_bias_frac &&
              SUM_HEAD >= q::seg_head_0_bias_frac,
              "a bias has more fraction bits than its layer's sums");
static_assert(SUM_PROJ > ACT_FRAC && SUM_CONV > ACT_FRAC && SUM_HEAD >= OUT_FRAC,
              "a requantization would have to add bits");

void waveletcnn_opt(hls::burst_maxi<wopt_raw_t> din0, hls::burst_maxi<wopt_raw_t> din1,
                    hls::burst_maxi<wopt_raw_t> din2, hls::burst_maxi<wopt_raw_t> din3,
                    hls::burst_maxi<wopt_out_t> dout, int height, int width, uint32_t& cycles)
{
    // Five AXI masters, each on its own port to DDR (boards/zcu104/system.tcl):
    // four read the cube in parallel (point all of them at it) through HP0-3,
    // one writes the scores through HPC0.
    // depth is only for co-simulation: the testbench's 40 x 50 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem4 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=cycles bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<wopt_raw_t> s_raw[PORTS];
    hls::stream<vec_t<act8_t, P1> > s_in("s_in");
    hls::stream<vec_t<act_t, MID_CH> > s_proj("s_proj");
    hls::stream<vec_t<acc_t, MID_CH> > s_conv("s_conv");
    hls::stream<vec_t<acc_t, OUT_CH> > s_head("s_head");
    hls::stream<bool> s_finished("s_finished");

    #pragma HLS STREAM variable=s_raw  depth=64
    #pragma HLS STREAM variable=s_in   depth=64
    #pragma HLS STREAM variable=s_proj depth=64
    #pragma HLS STREAM variable=s_conv depth=32
    #pragma HLS STREAM variable=s_head depth=32

    read_rows_ahead<H, W, WORDS, PORTS, 0>(din0, s_raw[0], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 1>(din1, s_raw[1], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 2>(din2, s_raw[2], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 3>(din3, s_raw[3], height, width);

    pixel_buffer<H, W, WORDS, FIRST, PORT_BANDS, P1, PORTS, DWT_STEPS,
                 prep::FRAC, prep::IN_MAX, prep::A, prep::B, true>(s_raw, s_in, height, width);

    // dwt_proj: 8 DWT steps/cycle (16 bands/cycle -> 8 cycles/pixel, 8 192 cycles/patch).
    // DWT filter taps in LUTs (8 fixed constants), projection 226 -> 12 in DSPs (384 DSPs).
    dwt_proj<H, W, IN_CH, DWT_LEN, MID_CH, M_PAR,
             q::dwt_spectral_h0_frac, SUM_PROJ - q::projection_bias_frac, SUM_PROJ - ACT_FRAC,
             q::dwt_spectral_h0, q::dwt_spectral_h1, q::projection_weight, q::projection_bias,
             false, true>(s_in, s_proj, height, width);

    // conv_block: 12 -> 12 3x3 same-padded, 2 output channels/cycle (6 cycles/pixel,
    // 6 342 cycles/patch, 216 DSPs).
    conv3x3_narrow<H, W, H, W, MID_CH, MID_CH, SUM_CONV - q::conv_block_0_bias_frac,
                   q::conv_block_0_weight, q::conv_block_0_bias, true, CONV_STEPS>(
        s_proj, s_conv, height, width);

    // leaky_head: LeakyReLU(0.01) on conv_block + 1x1 seg_head (12 -> 3) + LeakyReLU(0.01),
    // 1 cycle/pixel (36 DSPs).
    leaky_head<H, W, MID_CH, OUT_CH,
               SUM_CONV - ACT_FRAC, SUM_HEAD - q::seg_head_0_bias_frac, SUM_HEAD - OUT_FRAC,
               q::seg_head_0_weight, q::seg_head_0_bias, true>(s_conv, s_head, height, width);

    store_scores<H, W, 0>(s_head, dout, height, width, s_finished);
    count_cycles(s_finished, cycles);
}
