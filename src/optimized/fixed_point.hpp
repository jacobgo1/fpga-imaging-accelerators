#pragma once

#include <stdint.h>
#include "../common/hls_stream_shim.hpp"

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

static inline act_t max2(act_t a, act_t b) {
    #pragma HLS INLINE
    return a > b ? a : b;
}
