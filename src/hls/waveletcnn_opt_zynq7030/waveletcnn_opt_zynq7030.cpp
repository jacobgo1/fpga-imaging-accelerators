#include "waveletcnn_opt_zynq7030.hpp"
#include "../waveletcnn_opt/waveletcnn_prep.hpp"
#include "../../optimized/shared/read_rows.hpp"
#include "../../optimized/shared/pixel_buffer.hpp"
#include "../../optimized/shared/dwt_proj.hpp"
#include "../../optimized/shared/conv3x3_narrow.hpp"
#include "../../optimized/shared/leaky_head.hpp"
#include "../../optimized/shared/store_scores.hpp"
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

constexpr int WORDS = RAW_BANDS / PORT_BANDS;           // 30 words per pixel in the cube
constexpr int FIRST = prep::FIRST_BAND / PORT_BANDS;    // words 0-1 (raw bands 0-7) are dropped
constexpr int KEPT = WORDS - FIRST;                      // words 2-29: kept bands 0-111
static_assert(RAW_BANDS % PORT_BANDS == 0 && P1 % PORT_BANDS == 0, "a chunk is whole words");
static_assert(prep::FIRST_BAND % PORT_BANDS == 0 &&
              sizeof(prep::A) / sizeof(prep::A[0]) == KEPT * PORT_BANDS,
              "the kept bands must be whole words, to the end of each pixel");
static_assert(OUT_CH + 1 == 4, "a pixel of dout is 3 scores and the class");

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

void waveletcnn_opt_zynq7030(hls::burst_maxi<wopt_raw_t> din0, hls::burst_maxi<wopt_raw_t> din1,
                             hls::burst_maxi<wopt_raw_t> din2, hls::burst_maxi<wopt_raw_t> din3,
                             hls::burst_maxi<wopt_out_t> dout, int height, int width)
{
    // Five AXI masters on the four 64-bit HP ports (boards/zynq7030/system.tcl):
    // four read the cube in parallel through HP0-3, one writes the scores
    // sharing HP3 with the fourth reader.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=2
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=2
    #pragma HLS INTERFACE mode=m_axi port=din2 bundle=gmem2 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=2
    #pragma HLS INTERFACE mode=m_axi port=din3 bundle=gmem3 offset=slave depth=60000 max_read_burst_length=256 num_read_outstanding=2
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem4 offset=slave depth=2000 max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din2   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din3   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<wopt_raw_t> s_raw[PORTS];
    hls::stream<vec_t<act8_t, P1> > s_in("s_in");
    hls::stream<vec_t<act_t, MID_CH> > s_proj("s_proj");
    hls::stream<vec_t<acc_t, MID_CH> > s_conv("s_conv");
    hls::stream<vec_t<acc_t, OUT_CH> > s_head("s_head");

    #pragma HLS STREAM variable=s_raw  depth=64
    #pragma HLS STREAM variable=s_in   depth=64
    #pragma HLS STREAM variable=s_proj depth=64
    #pragma HLS STREAM variable=s_conv depth=32
    #pragma HLS STREAM variable=s_head depth=32

    read_rows_ahead<H, W, WORDS, PORTS, 0, 8, 2>(din0, s_raw[0], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 1, 8, 2>(din1, s_raw[1], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 2, 8, 2>(din2, s_raw[2], height, width);
    read_rows_ahead<H, W, WORDS, PORTS, 3, 8, 2>(din3, s_raw[3], height, width);

    pixel_buffer<H, W, WORDS, FIRST, PORT_BANDS, P1, PORTS, DWT_STEPS,
                 prep::FRAC, prep::IN_MAX, prep::A, prep::B, false>(s_raw, s_in, height, width);

    // dwt_proj: 4 DWT steps/cycle (8 bands/cycle -> 15 cycles/pixel, 15 360 cycles/patch).
    // DWT filter taps in LUTs, projection 226 -> 12 in DSPs (192 DSPs).
    dwt_proj<H, W, IN_CH, DWT_LEN, MID_CH, M_PAR,
             q::dwt_spectral_h0_frac, SUM_PROJ - q::projection_bias_frac, SUM_PROJ - ACT_FRAC,
             q::dwt_spectral_h0, q::dwt_spectral_h1, q::projection_weight, q::projection_bias,
             false, true>(s_in, s_proj, height, width);

    // conv_block: 12 -> 12 3x3 same-padded, 1 output channel/cycle (12 cycles/pixel,
    // 12 684 cycles/patch, 108 DSPs).
    conv3x3_narrow<H, W, H, W, MID_CH, MID_CH, SUM_CONV - q::conv_block_0_bias_frac,
                   q::conv_block_0_weight, q::conv_block_0_bias, true>(
        s_proj, s_conv, height, width);

    // leaky_head: 1 cycle/pixel (36 DSPs).
    leaky_head<H, W, MID_CH, OUT_CH,
               SUM_CONV - ACT_FRAC, SUM_HEAD - q::seg_head_0_bias_frac, SUM_HEAD - OUT_FRAC,
               q::seg_head_0_weight, q::seg_head_0_bias, true>(s_conv, s_head, height, width);

    store_scores<H, W, 0>(s_head, dout, height, width);
}
