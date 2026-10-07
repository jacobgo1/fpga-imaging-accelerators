#include "justounetsimple_opt.hpp"
#include "../../optimized/stream_ops.hpp"
#include <type_traits>

// The quantized weights, as the int8 integers q they are (value q * 2^-frac):
// this kernel multiplies integers. The scales become shifts, below.
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace q = justounetsimple_int8;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::conv1_weight)>::type>::type, wgt_t>::value,
              "the weights must be int8; did the header's element type change?");

constexpr int H = JOPT_H, W = JOPT_W, IN = JOPT_IN_CH, B = JOPT_BASE_CH, OUT = JOPT_OUT_CH;
constexpr int P1 = JOPT_P1, INP = JOPT_IN_PAD, PB = JOPT_PORT_BANDS;
static_assert(P1 == 2 * PB, "conv1's bands per cycle come from two ports");

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
// int32 never overflows: the largest |sum| any int16 input can make is
// 32767 * (the largest sum of |q| over one output channel), 3.7e8 for conv1.
// The testbench checks that bound against the real weights.

// ---- load: DDR -> stream, 16 bands per cycle, 8 from each port ----
static void load(const jopt_in_t *din0, const jopt_in_t *din1,
                 hls::stream<vec_t<act_t, P1> >& out, int n) {
    load: for (int i = 0; i < n * (INP / P1) * H * W; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=7168 max=458752
        const jopt_in_t lo = din0[i], hi = din1[i];
        vec_t<act_t, P1> v;
        for (int p = 0; p < PB; p++) {
            #pragma HLS UNROLL
            v.v[p] = lo[p];
            v.v[PB + p] = hi[p];
        }
        out.write(v);
    }
}

// ---- block A: conv1 (110 -> 6) at 32 x 32, 16 bands per cycle ----
// Reads 112 bands (110 + 2 zero bands, zero weights). 6 x 16 x 9 = 864
// multiply-adds per cycle.
static void conv_a(hls::stream<vec_t<act_t, P1> >& in, hls::stream<vec_t<acc_t, B> >& out, int n) {
    wgt_t wl[B][INP][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wl type=cyclic factor=P1 dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[B];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete

    load_conv_weights<B, IN, INP>(q::conv1_weight, q::conv1_bias, SUM1 - q::conv1_bias_frac, wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H, W, INP, B, P1, 1>(in, wl, bl, out);
    }
}

// ReLU, max-pool -> 16 x 16 x 6.
static void pool_a(hls::stream<vec_t<acc_t, B> >& in, hls::stream<vec_t<act_t, B> >& out, int n) {
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        relu_pool<H, W, B, SUM1 - JOPT_ACT_FRAC>(in, out);
    }
}

// The layers after conv1 take a pixel's channels all at once, and compute one
// output channel per cycle (STEPS = their output channels). Their weights: one
// small memory per (input channel, tap), addressed by the output channel.

// ---- block B: conv2 (6 -> 12) at 16 x 16, 12 cycles per pixel ----
static void conv_b(hls::stream<vec_t<act_t, B> >& in, hls::stream<vec_t<acc_t, 2 * B> >& out, int n) {
    wgt_t wl[2 * B][B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[2 * B];
    load_conv_weights<2 * B, B>(q::conv2_weight, q::conv2_bias, SUM2 - q::conv2_bias_frac, wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H / 2, W / 2, B, 2 * B, B, 2 * B>(in, wl, bl, out);
    }
}

// ReLU, max-pool, upsample -> 16 x 16 x 12.
static void pool_up_b(hls::stream<vec_t<acc_t, 2 * B> >& in, hls::stream<vec_t<act_t, 2 * B> >& out, int n) {
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        relu_pool_up<H / 2, W / 2, 2 * B, SUM2 - JOPT_ACT_FRAC>(in, out);
    }
}

// ---- block C: conv3 (12 -> 6) at 16 x 16, 6 cycles per pixel ----
static void conv_c(hls::stream<vec_t<act_t, 2 * B> >& in, hls::stream<vec_t<acc_t, B> >& out, int n) {
    wgt_t wl[B][2 * B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[B];
    load_conv_weights<B, 2 * B>(q::conv3_weight, q::conv3_bias, SUM3 - q::conv3_bias_frac, wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H / 2, W / 2, 2 * B, B, 2 * B, B>(in, wl, bl, out);
    }
}

// ReLU, upsample -> 32 x 32 x 6.
static void up_c(hls::stream<vec_t<acc_t, B> >& in, hls::stream<vec_t<act_t, B> >& out, int n) {
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        relu_up<H / 2, W / 2, B, SUM3 - JOPT_ACT_FRAC>(in, out);
    }
}

// ---- block D: conv4 (6 -> 3) at 32 x 32, no activation, 3 cycles per pixel ----
static void conv_d(hls::stream<vec_t<act_t, B> >& in, hls::stream<vec_t<acc_t, OUT> >& out, int n) {
    wgt_t wl[OUT][B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[OUT];
    load_conv_weights<OUT, B>(q::conv4_weight, q::conv4_bias, SUM4 - q::conv4_bias_frac, wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H, W, B, OUT, B, OUT>(in, wl, bl, out);
    }
}

// ---- store: stream -> DDR, a pixel's 3 scores at OUT_FRAC fraction bits ----
static void store(hls::stream<vec_t<acc_t, OUT> >& in, int32_t *dout, int n) {
    store: for (int i = 0; i < n * H * W; i++) {
        #pragma HLS PIPELINE II=OUT
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=65536
        const vec_t<acc_t, OUT> v = in.read();
        for (int c = 0; c < OUT; c++) {
            #pragma HLS UNROLL
            dout[i * OUT + c] = round_shift<SUM4 - JOPT_OUT_FRAC>(v.v[c]);
        }
    }
}

void justounetsimple_opt(const jopt_in_t *din0, const jopt_in_t *din1, int32_t *dout, int n)
{
    // Three AXI masters, each on its own HP port (boards/zcu104/system.tcl):
    // two read the input in parallel, one writes the scores. The PS sets the
    // DDR addresses and the patch count in the s_axilite registers. depth is
    // only for co-simulation: two patches, as in the testbench.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=14336 max_read_burst_length=64
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=14336 max_read_burst_length=64
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem2 offset=slave depth=6144 max_write_burst_length=64
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=n      bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<vec_t<act_t, P1> > s_in("s_in");
    hls::stream<vec_t<acc_t, B> > s_conv1("s_conv1");
    hls::stream<vec_t<act_t, B> > s_a("s_a");
    hls::stream<vec_t<acc_t, 2 * B> > s_conv2("s_conv2");
    hls::stream<vec_t<act_t, 2 * B> > s_b("s_b");
    hls::stream<vec_t<acc_t, B> > s_conv3("s_conv3");
    hls::stream<vec_t<act_t, B> > s_c("s_c");
    hls::stream<vec_t<acc_t, OUT> > s_conv4("s_conv4");
    // conv1 hands out a patch's 16 x 16 pooled pixels during its last chunk
    // only (1024 cycles), conv2 takes one per 12 cycles. Room for all of them,
    // so conv1 never waits for conv2.
    #pragma HLS STREAM variable=s_a depth=256

    load(din0, din1, s_in, n);
    conv_a(s_in, s_conv1, n);
    pool_a(s_conv1, s_a, n);
    conv_b(s_a, s_conv2, n);
    pool_up_b(s_conv2, s_b, n);
    conv_c(s_b, s_conv3, n);
    up_c(s_conv3, s_c, n);
    conv_d(s_c, s_conv4, n);
    store(s_conv4, dout, n);
}
