#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"

// The numbers every block in src/optimized/ shares, in integer fixed point.
// An activation is an int16 a meaning a * 2^-F (F fraction bits, set by the
// kernel), a weight is the quantized int8 q meaning q * 2^-frac, so a product
// is exact in an int32 with F + frac fraction bits, and so is a sum of them.
// No floating point anywhere: a 16 x 8 bit multiply is one DSP, against about
// five for a float multiply-add.

typedef int16_t act_t;   // an activation (fixed point, F fraction bits)
typedef int8_t  act8_t;  // an int8 activation, -127 .. 127 (conv3x3_wide's input)
typedef int8_t  wgt_t;   // a quantized weight q
typedef int32_t acc_t;   // a sum of products, and a bias at the same scale

// P values that travel together through a stream (P channels of one pixel).
template<typename T, int P>
struct vec_t {
    T v[P];
};

// One sum -> one activation: ReLU, round off SHIFT bits, saturate at the int16
// maximum. Rounding is monotonic, so max-pooling before or after it gives the
// same result.
template<int SHIFT>
act_t requant_relu(acc_t v) {
    #pragma HLS INLINE
    static_assert(SHIFT >= 1, "requant_relu drops at least one bit");
    if (v <= 0) return 0;
    const acc_t r = (v + ((acc_t)1 << (SHIFT - 1))) >> SHIFT;
    return r > 32767 ? (act_t)32767 : (act_t)r;
}

// Round off SHIFT bits (no ReLU, no saturation: the result only gets smaller).
template<int SHIFT>
acc_t round_shift(acc_t v) {
    #pragma HLS INLINE
    static_assert(SHIFT >= 0, "round_shift only drops bits");
    return SHIFT == 0 ? v : (v + ((acc_t)1 << (SHIFT > 0 ? SHIFT - 1 : 0))) >> SHIFT;
}

// One sum -> one signed int16 activation (no ReLU): round off SHIFT bits,
// saturate to [-32768, 32767].
template<int SHIFT>
act_t requant_act(acc_t v) {
    #pragma HLS INLINE
    const acc_t r = round_shift<SHIFT>(v);
    return r > 32767 ? (act_t)32767 : r < -32768 ? (act_t)-32768 : (act_t)r;
}

// LeakyReLU in integers: v >= 0 ? v : round_shift<LEAK_FRAC>(v * LEAK_NUM).
// Default 41 * 2^-12 = 41 / 4096 = 0.01000977 (PyTorch's 0.01 negative slope);
// 41 = 32 + 8 + 1 is a constant, so HLS turns it into two additions in LUTs.
template<int LEAK_NUM = 41, int LEAK_FRAC = 12>
acc_t leaky_relu(acc_t v) {
    #pragma HLS INLINE
    if (v >= 0) return v;
    const int64_t p = (int64_t)v * LEAK_NUM + ((int64_t)1 << (LEAK_FRAC - 1));
    return (acc_t)(p >> LEAK_FRAC);
}

// One sum -> one signed int16 activation with LeakyReLU(0.01): round off SHIFT
// bits, apply leaky_relu, saturate to [-32768, 32767].
template<int SHIFT, int LEAK_NUM = 41, int LEAK_FRAC = 12>
act_t requant_leaky_relu(acc_t v) {
    #pragma HLS INLINE
    const acc_t r = leaky_relu<LEAK_NUM, LEAK_FRAC>(round_shift<SHIFT>(v));
    return r > 32767 ? (act_t)32767 : r < -32768 ? (act_t)-32768 : (act_t)r;
}

// What a conv layer sends on: with RELU_SHIFT 0 its sums as they are (for a
// ReLU module after it, or a final layer's scores); otherwise each ReLU'd and
// rounded off RELU_SHIFT bits to an int16 activation, the next conv's input.
template<int RELU_SHIFT> struct sums_out {
    typedef act_t type;
    static type of(acc_t v) {
        #pragma HLS INLINE
        return requant_relu<RELU_SHIFT>(v);
    }
};
template<> struct sums_out<0> {
    typedef acc_t type;
    static type of(acc_t v) {
        #pragma HLS INLINE
        return v;
    }
};

static inline act_t max2(act_t a, act_t b) {
    #pragma HLS INLINE
    return a > b ? a : b;
}

// a * b in R, built from a DSP (DSP = true) or from LUTs (false; HLS calls
// them "fabric"), said with a pragma instead of left to HLS. BIND_OP's impl
// takes only a literal word, not a template argument, so each choice has its
// own branch; HLS keeps the one DSP picks. The modules with multipliers take
// DSP as a template argument, so a kernel's top function shows where each
// process's go; the index arithmetic (patch counts, row addresses) is always
// LUTs. The csynth report's "Bind Op Report" lists what HLS built. A multiply
// by a constant needs neither: HLS turns it into shifts and adds.
template<bool DSP, typename R, typename A, typename B>
R mul(A a, B b) {
    #pragma HLS INLINE
    R p;
    if (DSP) {
        const R d = (R)a * (R)b;
        #pragma HLS BIND_OP variable=d op=mul impl=dsp
        p = d;
    } else {
        const R f = (R)a * (R)b;
        #pragma HLS BIND_OP variable=f op=mul impl=fabric
        p = f;
    }
    return p;
}
