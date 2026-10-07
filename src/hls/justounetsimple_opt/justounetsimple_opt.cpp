#include "justounetsimple_opt.hpp"
#include "justounetsimple_prep.hpp"
#include "../../optimized/stream_ops.hpp"
#include <type_traits>

// The quantized weights, as the int8 integers q they are (value q * 2^-frac):
// this kernel multiplies integers. The scales become shifts, below.
#include "../../../weights/quantized/justounetsimple/justounetsimple_int8.hpp"

namespace q = justounetsimple_int8;
namespace prep = justounetsimple_prep;
static_assert(std::is_same<std::remove_const<std::remove_all_extents<
                  decltype(q::conv1_weight)>::type>::type, wgt_t>::value,
              "the weights must be int8; did the header's element type change?");
static_assert(prep::IN_FRAC == JOPT_IN_FRAC, "justounetsimple_prep.hpp is out of date: "
              "run python tools/justounetsimple_model.py");

constexpr int H = JOPT_H, W = JOPT_W, IN = JOPT_IN_CH, B = JOPT_BASE_CH, OUT = JOPT_OUT_CH;
constexpr int P1 = JOPT_P1, INP = JOPT_IN_PAD, PB = JOPT_PORT_BANDS;
constexpr int WORDS = JOPT_RAW_BANDS / PB;      // 15 words per pixel in the cube
constexpr int CHUNKS = INP / P1;                // 7 chunks of 16 bands for conv1
static_assert(JOPT_RAW_BANDS % PB == 0 && P1 == 2 * PB, "a chunk is two words");
static_assert(prep::FIRST_BAND == PB && (CHUNKS * 2 + 1) * PB <= JOPT_RAW_BANDS,
              "the kept bands must start at word 1 of each pixel");

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

// The image is cut into patch rows x patch columns, raster order (loop tiling:
// every process below walks the same grid).
static int patch_rows(int height) { return (height + H - 1) / H; }
static int patch_cols(int width) { return (width + W - 1) / W; }

// ---- read_rows: DDR -> stream, the raw words of every other row of each patch ----
// For each patch, rows PORT, PORT + 2, ... : 32 pixels x 15 words each, in
// cube order. One burst per row. Past the image's bottom the last row is read
// again; past its right edge the last pixel is repeated, so every patch is
// whole (the same as the notebook's edge padding used to be).
template<int PORT>
static void read_rows(const jopt_raw_t *din, hls::stream<jopt_raw_t>& out, int height, int width) {
    const int rows = patch_rows(height), cols = patch_cols(width);
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int i = 0; i < H / 2; i++) {
                const int y = pr * H + 2 * i + PORT;
                const int x0 = pc * W;
                const int n = width - x0 < W ? width - x0 : W;   // pixels inside the image
                const jopt_raw_t *src = din + ((uint32_t)(y < height ? y : height - 1) * width + x0) * WORDS;
                jopt_raw_t last[WORDS];                          // the last pixel read
                #pragma HLS ARRAY_PARTITION variable=last type=complete
                int w = 0;
                take: for (int k = 0; k < n * WORDS; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=15 max=480
                    const jopt_raw_t v = src[k];
                    out.write(v);
                    last[w] = v;
                    w = (w == WORDS - 1) ? 0 : w + 1;
                }
                edge: for (int k = 0; k < (W - n) * WORDS; k++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=465
                    out.write(last[w]);
                    w = (w == WORDS - 1) ? 0 : w + 1;
                }
            }
        }
    }
}

// ---- patch_buffer: z-score, then the ping-pong between the readers and conv1 ----
// Round k fills patch k into one buffer while it sends patch k-1 from the
// other to conv1, 16 bands of a pixel per cycle in conv1's order (chunk by
// chunk, each in raster order); then the buffers swap. n patches take n + 1
// rounds of 7680 cycles (the words of one patch per port).
constexpr int FILL = H / 2 * W * WORDS;         // words per port per patch
constexpr int EMIT = CHUNKS * H * W;            // vectors per patch to conv1
static_assert(FILL >= EMIT, "a round must have time to send a whole patch");

// One bank of the patch buffers: both ping-pong buffers, [buffer][chunk][row / 2][x],
// flattened. 8 bands (128 bits) per entry.
constexpr int BANK_HALF = CHUNKS * (H / 2) * W;
typedef vec_t<act_t, PB> bank_t[2 * BANK_HALF];

static void patch_buffer(hls::stream<jopt_raw_t>& raw0, hls::stream<jopt_raw_t>& raw1,
                         hls::stream<vec_t<act_t, P1> >& out, int height, int width) {
    // Four banks: [row parity = port][word of the chunk]. The ping-pong buffer is
    // the high half of the address, not a bank of its own. Each bank has exactly
    // one write and one read in the loop below, at fixed places: a bank whose
    // write or read HLS has to pick at run time gets one store per possible
    // pick, and its single write port then takes that many cycles (II = 5 when
    // the buffer and the word half were array indices).
    bank_t b00, b01, b10, b11;
    #pragma HLS AGGREGATE variable=b00 compact=bit
    #pragma HLS AGGREGATE variable=b01 compact=bit
    #pragma HLS AGGREGATE variable=b10 compact=bit
    #pragma HLS AGGREGATE variable=b11 compact=bit
    #pragma HLS BIND_STORAGE variable=b00 type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=b01 type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=b10 type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=b11 type=ram_2p impl=uram
    // The z-score constants of the kept words (1-14), one small ROM per lane.
    int32_t za[2 * CHUNKS][PB], zb[2 * CHUNKS][PB];
    #pragma HLS ARRAY_PARTITION variable=za type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=zb type=complete dim=2
    load_z: for (int k = 0; k < 2 * CHUNKS * PB; k++) {
        #pragma HLS PIPELINE II=1
        za[k / PB][k % PB] = prep::A[k];
        zb[k / PB][k % PB] = prep::B[k];
    }

    const int n = patch_rows(height) * patch_cols(width);
    int wb = 0;                                 // the buffer being filled
    round: for (int k = 0; k <= n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=2 max=666
        const bool fill = k < n, emit = k > 0;
        int fr = 0, fx = 0, fw = 0;             // fill: row pair, pixel, word
        int ec = 0, ey = 0, ex = 0;             // emit: chunk, row, pixel
        step: for (int t = 0; t < FILL; t++) {
            #pragma HLS PIPELINE II=1
            // Writes go to one half of each bank, reads to the other.
            #pragma HLS DEPENDENCE variable=b00 type=inter false
            #pragma HLS DEPENDENCE variable=b01 type=inter false
            #pragma HLS DEPENDENCE variable=b10 type=inter false
            #pragma HLS DEPENDENCE variable=b11 type=inter false
            #pragma HLS DEPENDENCE variable=b00 type=intra false
            #pragma HLS DEPENDENCE variable=b01 type=intra false
            #pragma HLS DEPENDENCE variable=b10 type=intra false
            #pragma HLS DEPENDENCE variable=b11 type=intra false
            jopt_raw_t w0, w1;
            if (fill) {
                w0 = raw0.read();
                w1 = raw1.read();
            }
            // Word 0 (raw bands 0-7) is dropped; kept word kw goes to chunk kw / 2,
            // into the bank of half kw % 2.
            const int kw = fw > 0 ? fw - 1 : 0;
            const bool store = fill && fw > 0;
            vec_t<act_t, PB> z0, z1;
            for (int p = 0; p < PB; p++) {
                #pragma HLS UNROLL
                // round((dn - mean) / std * 2^IN_FRAC), saturated
                const int64_t s0 = ((int64_t)w0[p] * za[kw][p] + zb[kw][p]) >> prep::FRAC;
                const int64_t s1 = ((int64_t)w1[p] * za[kw][p] + zb[kw][p]) >> prep::FRAC;
                z0.v[p] = (act_t)(s0 > 32767 ? 32767 : s0 < -32768 ? -32768 : s0);
                z1.v[p] = (act_t)(s1 > 32767 ? 32767 : s1 < -32768 ? -32768 : s1);
            }
            const int waddr = (wb ? BANK_HALF : 0) + ((kw >> 1) * (H / 2) + fr) * W + fx;
            if (store && (kw & 1) == 0) b00[waddr] = z0;
            if (store && (kw & 1) == 1) b01[waddr] = z0;
            if (store && (kw & 1) == 0) b10[waddr] = z1;
            if (store && (kw & 1) == 1) b11[waddr] = z1;
            if (fill) {
                if (fw < WORDS - 1) fw++;
                else { fw = 0; if (fx < W - 1) fx++; else { fx = 0; fr++; } }
            }

            if (emit && t < EMIT) {
                // All four banks are read; the row's parity picks the pair.
                const int raddr = (wb ? 0 : BANK_HALF) + (ec * (H / 2) + (ey >> 1)) * W + ex;
                const vec_t<act_t, PB> r00 = b00[raddr], r01 = b01[raddr];
                const vec_t<act_t, PB> r10 = b10[raddr], r11 = b11[raddr];
                const vec_t<act_t, PB> lo = (ey & 1) ? r10 : r00;
                const vec_t<act_t, PB> hi = (ey & 1) ? r11 : r01;
                vec_t<act_t, P1> v;
                for (int p = 0; p < PB; p++) {
                    #pragma HLS UNROLL
                    v.v[p] = lo.v[p];
                    v.v[PB + p] = hi.v[p];
                }
                out.write(v);
                if (ex < W - 1) ex++;
                else { ex = 0; if (ey < H - 1) ey++; else { ey = 0; ec++; } }
            }
        }
        wb = 1 - wb;
    }
}

// ---- block A: conv1 (110 -> 6) at 32 x 32, 16 bands per cycle ----
// Reads 112 bands (110 + 2 zero bands, zero weights). 6 x 16 x 9 = 864
// multiply-adds per cycle.
static void conv_a(hls::stream<vec_t<act_t, P1> >& in, hls::stream<vec_t<acc_t, B> >& out,
                   int height, int width) {
    wgt_t wl[B][INP][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wl type=cyclic factor=P1 dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[B];
    #pragma HLS ARRAY_PARTITION variable=bl type=complete

    load_conv_weights<B, IN, INP>(q::conv1_weight, q::conv1_bias, SUM1 - q::conv1_bias_frac, wl, bl);
    const int n = patch_rows(height) * patch_cols(width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_stream<H, W, INP, B, P1, 1>(in, wl, bl, out);
    }
}

// ReLU, max-pool -> 16 x 16 x 6.
static void pool_a(hls::stream<vec_t<acc_t, B> >& in, hls::stream<vec_t<act_t, B> >& out,
                   int height, int width) {
    const int n = patch_rows(height) * patch_cols(width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_pool<H, W, B, SUM1 - JOPT_ACT_FRAC>(in, out);
    }
}

// The layers after conv1 take a pixel's channels all at once, and compute one
// output channel per cycle (STEPS = their output channels). Their weights: one
// small memory per (input channel, tap), addressed by the output channel.

// ---- block B: conv2 (6 -> 12) at 16 x 16, 12 cycles per pixel ----
static void conv_b(hls::stream<vec_t<act_t, B> >& in, hls::stream<vec_t<acc_t, 2 * B> >& out,
                   int height, int width) {
    wgt_t wl[2 * B][B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[2 * B];
    load_conv_weights<2 * B, B>(q::conv2_weight, q::conv2_bias, SUM2 - q::conv2_bias_frac, wl, bl);
    const int n = patch_rows(height) * patch_cols(width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_stream<H / 2, W / 2, B, 2 * B, B, 2 * B>(in, wl, bl, out);
    }
}

// ReLU, max-pool, upsample -> 16 x 16 x 12.
static void pool_up_b(hls::stream<vec_t<acc_t, 2 * B> >& in, hls::stream<vec_t<act_t, 2 * B> >& out,
                      int height, int width) {
    const int n = patch_rows(height) * patch_cols(width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_pool_up<H / 2, W / 2, 2 * B, SUM2 - JOPT_ACT_FRAC>(in, out);
    }
}

// ---- block C: conv3 (12 -> 6) at 16 x 16, 6 cycles per pixel ----
static void conv_c(hls::stream<vec_t<act_t, 2 * B> >& in, hls::stream<vec_t<acc_t, B> >& out,
                   int height, int width) {
    wgt_t wl[B][2 * B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[B];
    load_conv_weights<B, 2 * B>(q::conv3_weight, q::conv3_bias, SUM3 - q::conv3_bias_frac, wl, bl);
    const int n = patch_rows(height) * patch_cols(width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_stream<H / 2, W / 2, 2 * B, B, 2 * B, B>(in, wl, bl, out);
    }
}

// ReLU, upsample -> 32 x 32 x 6.
static void up_c(hls::stream<vec_t<acc_t, B> >& in, hls::stream<vec_t<act_t, B> >& out,
                 int height, int width) {
    const int n = patch_rows(height) * patch_cols(width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        relu_up<H / 2, W / 2, B, SUM3 - JOPT_ACT_FRAC>(in, out);
    }
}

// ---- block D: conv4 (6 -> 3) at 32 x 32, no activation, 3 cycles per pixel ----
static void conv_d(hls::stream<vec_t<act_t, B> >& in, hls::stream<vec_t<acc_t, OUT> >& out,
                   int height, int width) {
    wgt_t wl[OUT][B][3][3];
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=2
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
    #pragma HLS ARRAY_PARTITION variable=wl type=complete dim=4
    acc_t bl[OUT];
    load_conv_weights<OUT, B>(q::conv4_weight, q::conv4_bias, SUM4 - q::conv4_bias_frac, wl, bl);
    const int n = patch_rows(height) * patch_cols(width);
    patches: for (int k = 0; k < n; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=665
        conv3x3_stream<H, W, B, OUT, B, OUT>(in, wl, bl, out);
    }
}

// ---- store: stream -> DDR, each patch row into its place in the image ----
// A patch row's scores (3 per pixel, OUT_FRAC fraction bits) are collected,
// then written in one burst -- only the pixels inside the image.
static void store(hls::stream<vec_t<acc_t, OUT> >& in, int32_t *dout, int height, int width) {
    const int rows = patch_rows(height), cols = patch_cols(width);
    patch_row: for (int pr = 0; pr < rows; pr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=19
        patch_col: for (int pc = 0; pc < cols; pc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=35
            line: for (int r = 0; r < H; r++) {
                int32_t scores[W * OUT];
                #pragma HLS ARRAY_PARTITION variable=scores type=cyclic factor=OUT
                take: for (int x = 0; x < W; x++) {
                    #pragma HLS PIPELINE II=1
                    const vec_t<acc_t, OUT> v = in.read();
                    for (int c = 0; c < OUT; c++) {
                        #pragma HLS UNROLL
                        scores[x * OUT + c] = round_shift<SUM4 - JOPT_OUT_FRAC>(v.v[c]);
                    }
                }
                const int y = pr * H + r, x0 = pc * W;
                const int n = width - x0 < W ? width - x0 : W;
                if (y < height) {
                    int32_t *dst = dout + ((uint32_t)y * width + x0) * OUT;
                    put: for (int k = 0; k < n * OUT; k++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=3 max=96
                        dst[k] = scores[k];
                    }
                }
            }
        }
    }
}

void justounetsimple_opt(const jopt_raw_t *din0, const jopt_raw_t *din1, int32_t *dout,
                         int height, int width)
{
    // Three AXI masters, each on its own HP port (boards/zcu104/system.tcl):
    // two read the cube in parallel (point both at it), one writes the scores.
    // depth is only for co-simulation: the testbench's 40 x 50 image.
    #pragma HLS INTERFACE mode=m_axi port=din0 bundle=gmem0 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=din1 bundle=gmem1 offset=slave depth=30000 max_read_burst_length=256 num_read_outstanding=4
    #pragma HLS INTERFACE mode=m_axi port=dout bundle=gmem2 offset=slave depth=6000 max_write_burst_length=128
    #pragma HLS INTERFACE mode=s_axilite port=din0   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=din1   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=dout   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=height bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=width  bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<jopt_raw_t> s_raw0("s_raw0"), s_raw1("s_raw1");
    hls::stream<vec_t<act_t, P1> > s_in("s_in");
    hls::stream<vec_t<acc_t, B> > s_conv1("s_conv1");
    hls::stream<vec_t<act_t, B> > s_a("s_a");
    hls::stream<vec_t<acc_t, 2 * B> > s_conv2("s_conv2");
    hls::stream<vec_t<act_t, 2 * B> > s_b("s_b");
    hls::stream<vec_t<acc_t, B> > s_conv3("s_conv3");
    hls::stream<vec_t<act_t, B> > s_c("s_c");
    hls::stream<vec_t<acc_t, OUT> > s_conv4("s_conv4");
    // The readers pause between rows (a new burst); let them run ahead.
    #pragma HLS STREAM variable=s_raw0 depth=64
    #pragma HLS STREAM variable=s_raw1 depth=64
    #pragma HLS STREAM variable=s_in depth=64
    // conv1 hands out a patch's 16 x 16 pooled pixels during its last chunk
    // only (1024 cycles), conv2 takes one per 12 cycles. Room for all of them,
    // so conv1 never waits for conv2.
    #pragma HLS STREAM variable=s_a depth=256
    // A patch row of scores, so conv4 keeps going while store writes one out.
    #pragma HLS STREAM variable=s_conv4 depth=32

    read_rows<0>(din0, s_raw0, height, width);
    read_rows<1>(din1, s_raw1, height, width);
    patch_buffer(s_raw0, s_raw1, s_in, height, width);
    conv_a(s_in, s_conv1, height, width);
    pool_a(s_conv1, s_a, height, width);
    conv_b(s_a, s_conv2, height, width);
    pool_up_b(s_conv2, s_b, height, width);
    conv_c(s_b, s_conv3, height, width);
    up_c(s_conv3, s_c, height, width);
    conv_d(s_c, s_conv4, height, width);
    store(s_conv4, dout, height, width);
}
