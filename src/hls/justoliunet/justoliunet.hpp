#ifndef JUSTOLIUNET_HPP
#define JUSTOLIUNET_HPP

#include "../../golden/justoliunet_golden.hpp"

// =====================================================================
// 1. What this kernel is, and why it looks this thin
//
//    src/golden/justoliunet_golden.hpp is already plain, synthesizable
//    C++ -- four (conv1d -> relu -> maxpool1d) stages widening
//    BASE_CH -> 2x -> 3x -> 4x while W shrinks, then flatten + one dense
//    layer, applied independently per row (see that header for the full
//    picture and the PyTorch model it mirrors). This kernel does nothing
//    but fix its template parameters to concrete geometry (part 3 below)
//    and forward the interface arrays straight through: no staging
//    buffers, no array partitioning, no pipelining beyond whatever HLS
//    infers on its own.
//
//    That is deliberate, not unfinished. This is the *baseline*: the
//    csynth resource/latency numbers this kernel produces before any
//    directive exists are what config/justoliunet.tcl's later pragmas
//    (see config/matmul.tcl for the pattern) should be measured against.
//    The same justoliunet_golden<float, ...> call, byte-for-byte, is also
//    what tb/justoliunet's testbench runs natively -- so `main.py native`
//    cross-compiled for the ZCU104's Cortex-A53 gives the matching
//    software-side number for the same computation.
//
//    Every array argument is an m_axi port sharing one AXI master
//    (justoliunet.cpp), not a BRAM port: the kernel DMAs its inputs
//    straight out of DDR and writes dout back to it, at whatever
//    addresses the PS puts in the matching s_axilite registers. That is
//    what makes feeding a real hyperspectral tile, and swapping in a
//    newly-trained model's weights, a plain host-side memory write
//    instead of a re-synthesized bitstream -- see software/zcu104/.
// =====================================================================

// =====================================================================
// 2. Data type
//
//    float, not a fixed-point type. This keeps every stage numerically
//    identical to a trained PyTorch state_dict and to the A53 baseline --
//    no quantization error of this kernel's own making, so a PL-vs-A53
//    comparison is apples to apples. Swapping data_t for an ap_fixed<>
//    (narrow on the weights, wide enough on accumulators not to overflow)
//    is the obvious first area/latency experiment once this float
//    baseline has a number -- see matmul.hpp's result_t comment for the
//    same idea applied there.
// =====================================================================
using data_t = float;

// =====================================================================
// 3. Layer geometry -- the only block you normally edit
//
//    #define, not constexpr: pragma arguments are parsed from
//    preprocessor text (see conv2d.hpp part 2). H is deliberately small --
//    a handful of spectra per call, not a whole HYPSO line -- because
//    cosim actually runs every one of these cycles through an RTL
//    simulator; make it bigger once you are past the "does this compile
//    and synthesize" stage. Retarget W from a real capture the same way
//    conv2d does, with tools/hypso_dims.py.
// =====================================================================
#define JNET_H              4   // spectra (rows) classified per call
#define JNET_W            110   // spectral bands per input spectrum
#define JNET_K              3   // conv1d kernel length
#define JNET_BASE_CH        8   // channel count after stage 1 (k in the golden header's docstring)
#define JNET_NUM_CLASSES    4   // classes the dense head scores

// =====================================================================
// 4. Derived per-stage geometry
//
//    Computed with justoliunet_golden.hpp's own constexpr helpers, not
//    reimplemented here -- see streaming_pipeline.hpp part 2 for why a
//    second copy of the same arithmetic is exactly how a module and its
//    caller quietly drift apart.
// =====================================================================
constexpr int JNET_W1C = justoliunet_conv_len(JNET_W,   JNET_K);
constexpr int JNET_W1P = justoliunet_pool_len(JNET_W1C);
constexpr int JNET_W2C = justoliunet_conv_len(JNET_W1P, JNET_K);
constexpr int JNET_W2P = justoliunet_pool_len(JNET_W2C);
constexpr int JNET_W3C = justoliunet_conv_len(JNET_W2P, JNET_K);
constexpr int JNET_W3P = justoliunet_pool_len(JNET_W3C);
constexpr int JNET_W4C = justoliunet_conv_len(JNET_W3P, JNET_K);
constexpr int JNET_W4P = justoliunet_pool_len(JNET_W4C);
constexpr int JNET_FLAT = justoliunet_flatten_dim(JNET_W, JNET_K, JNET_BASE_CH);

// =====================================================================
// 5. Compile-time sanity checks (see conv2d.hpp part 4)
//
//    The golden header's own comment promises a bad W/K combination
//    "fails to compile ... not silently misbehave", but that is only true
//    for a stage that hits zero or a negative array bound. An *odd* conv
//    output silently drops maxpool1d_golden's last column instead of
//    failing anything -- these asserts turn that gap into a real build
//    error here.
// =====================================================================
static_assert(JNET_W1C > 0 && JNET_W2C > 0 && JNET_W3C > 0 && JNET_W4C > 0,
             "kernel K is longer than one of the four stage inputs");
static_assert(JNET_W1C % 2 == 0 && JNET_W2C % 2 == 0 && JNET_W3C % 2 == 0 && JNET_W4C % 2 == 0,
             "a conv output length is odd: maxpool1d would silently drop its last column -- "
             "pick a different W/K so every stage halves exactly");
static_assert(JNET_W1P > 0 && JNET_W2P > 0 && JNET_W3P > 0 && JNET_W4P > 0,
             "W shrinks to zero before the fourth stage: widen W or shrink K");
static_assert(JNET_FLAT == 4 * JNET_BASE_CH * JNET_W4P,
             "justoliunet_flatten_dim and this header's manual stage math disagree");

// =====================================================================
// 6. Plain constexpr mirrors, for testbench / host code (see conv2d.hpp
//    part 5).
// =====================================================================
namespace jnet {
constexpr int H = JNET_H, W = JNET_W, K = JNET_K;
constexpr int base_ch = JNET_BASE_CH, num_classes = JNET_NUM_CLASSES;
constexpr int flat = JNET_FLAT;
}

void justoliunet(
    data_t din[JNET_H][JNET_W][1],
    data_t w1[JNET_BASE_CH][1][JNET_K],                    data_t b1[JNET_BASE_CH],
    data_t w2[2 * JNET_BASE_CH][JNET_BASE_CH][JNET_K],      data_t b2[2 * JNET_BASE_CH],
    data_t w3[3 * JNET_BASE_CH][2 * JNET_BASE_CH][JNET_K],  data_t b3[3 * JNET_BASE_CH],
    data_t w4[4 * JNET_BASE_CH][3 * JNET_BASE_CH][JNET_K],  data_t b4[4 * JNET_BASE_CH],
    data_t w5[JNET_NUM_CLASSES][JNET_FLAT],                 data_t b5[JNET_NUM_CLASSES],
    data_t dout[JNET_H][JNET_NUM_CLASSES]);

#endif // JUSTOLIUNET_HPP
