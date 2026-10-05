#include "justounetsimple_opt.hpp"
#include <type_traits>

// The quantized weights, read with the header's int8_t meaning float (see
// src/hls/justounetsimple/justounetsimple.cpp for why): float ROM holding the
// integers q, no int-to-float converter in hardware.
namespace justounetsimple_int8 { using int8_t = float; }
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace q = justounetsimple_int8;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::conv1_weight)>::type>::type, float>::value,
              "the weights must be read as float; did the header's element type change?");

constexpr float scale_of(int frac) { return 1.0f / (float)(1L << frac); }

constexpr int H = JOPT_H, W = JOPT_W, IN = JOPT_IN_CH, B = JOPT_BASE_CH, OUT = JOPT_OUT_CH;
constexpr int P1 = JOPT_P1;

// ---- load: DDR -> stream, one 2-band vector (64 bits) per cycle ----
static void load(const jopt_in_t *din, hls::stream<vec_t<P1> >& out, int n) {
    load: for (int i = 0; i < n * (IN / P1) * H * W; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=56320 max=3604480
        const jopt_in_t word = din[i];
        vec_t<P1> v;
        for (int p = 0; p < P1; p++) {
            #pragma HLS UNROLL
            v.v[p] = word[p];
        }
        out.write(v);
    }
}

// ---- block A: conv1 (110 -> 6) at 32 x 32, ReLU, max-pool -> 16 x 16 x 6 ----
static void block_a(hls::stream<vec_t<P1> >& in, hls::stream<vec_t<1> >& out, int n) {
    float wl[B][IN][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wl type=cyclic factor=P1 dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    float bl[B];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete
    float acc[H][W][B];
    #pragma HLS ARRAY_PARTITION variable=acc type=cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=acc type=cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=acc type=complete dim=3

    load_conv_weights<B, IN>(q::conv1_weight, scale_of(q::conv1_weight_frac),
                             q::conv1_bias, scale_of(q::conv1_bias_frac), wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H, W, IN, B, P1, 1>(in, wl, bl, acc);
        emit_relu_pool<H, W, B>(acc, out);
    }
}

// ---- block B: conv2 (6 -> 12) at 16 x 16, ReLU, max-pool, upsample -> 16 x 16 x 12 ----
static void block_b(hls::stream<vec_t<1> >& in, hls::stream<vec_t<1> >& out, int n) {
    float wl[2 * B][B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    float bl[2 * B];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete
    float acc[H / 2][W / 2][2 * B];
    #pragma HLS ARRAY_PARTITION variable=acc type=cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=acc type=cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=acc type=complete dim=3

    load_conv_weights<2 * B, B>(q::conv2_weight, scale_of(q::conv2_weight_frac),
                                q::conv2_bias, scale_of(q::conv2_bias_frac), wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H / 2, W / 2, B, 2 * B, 1, 4>(in, wl, bl, acc);
        emit_relu_pool_up<H / 2, W / 2, 2 * B>(acc, out);
    }
}

// ---- block C: conv3 (12 -> 6) at 16 x 16, ReLU, upsample -> 32 x 32 x 6 ----
static void block_c(hls::stream<vec_t<1> >& in, hls::stream<vec_t<1> >& out, int n) {
    float wl[B][2 * B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    float bl[B];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete
    float acc[H / 2][W / 2][B];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete dim=3

    load_conv_weights<B, 2 * B>(q::conv3_weight, scale_of(q::conv3_weight_frac),
                                q::conv3_bias, scale_of(q::conv3_bias_frac), wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H / 2, W / 2, 2 * B, B, 1, 4>(in, wl, bl, acc);
        emit_relu_up<H / 2, W / 2, B>(acc, out);
    }
}

// ---- block D: conv4 (6 -> 3) at 32 x 32, no activation -> 32 x 32 x 3 scores ----
static void block_d(hls::stream<vec_t<1> >& in, hls::stream<vec_t<OUT> >& out, int n) {
    float wl[OUT][B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    float bl[OUT];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete
    float acc[H][W][OUT];
    #pragma HLS ARRAY_PARTITION variable=acc type=complete dim=3

    load_conv_weights<OUT, B>(q::conv4_weight, scale_of(q::conv4_weight_frac),
                              q::conv4_bias, scale_of(q::conv4_bias_frac), wl, bl);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        conv3x3_stream<H, W, B, OUT, 1, 2>(in, wl, bl, acc);
        emit_pixels<H, W, OUT>(acc, out);
    }
}

// ---- store: stream -> DDR, one pixel's 3 scores at a time ----
static void store(hls::stream<vec_t<OUT> >& in, float *dout, int n) {
    store: for (int i = 0; i < n * H * W; i++) {
        #pragma HLS PIPELINE II=OUT
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=65536
        vec_t<OUT> v = in.read();
        for (int c = 0; c < OUT; c++) {
            #pragma HLS UNROLL
            dout[i * OUT + c] = v.v[c];
        }
    }
}

void justounetsimple_opt(const jopt_in_t *din, float *dout, int n)
{
    // Two AXI masters (reads and writes in parallel); the PS sets the DDR
    // addresses and the patch count in the s_axilite registers. depth is only
    // for co-simulation: two patches, as in the testbench.
    #pragma HLS INTERFACE mode=m_axi port=din  bundle=gmem0 offset=slave depth=112640 max_read_burst_length=64
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem1 offset=slave depth=6144 max_write_burst_length=64
    #pragma HLS INTERFACE mode=s_axilite port=din    bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=n      bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<vec_t<P1> > s_in("s_in");
    hls::stream<vec_t<1> > s_a("s_a"), s_b("s_b"), s_c("s_c");
    hls::stream<vec_t<OUT> > s_d("s_d");

    load(din, s_in, n);
    block_a(s_in, s_a, n);
    block_b(s_a, s_b, n);
    block_c(s_b, s_c, n);
    block_d(s_c, s_d, n);
    store(s_d, dout, n);
}
