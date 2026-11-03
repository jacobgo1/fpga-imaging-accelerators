#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"

// Integer fixed point, shared by every module. An activation a means a * 2^-F,
// a weight q means q * 2^-frac, so products and their sums are exact in int32.

typedef int16_t act_t;   // an activation
typedef int8_t  act8_t;  // an int8 activation, -127 .. 127
typedef int8_t  weight_t;   // a quantized weight
typedef int32_t sum_t;   // a sum of products, or a bias at the same scale

// SIZE values that travel together through a stream, e.g. the channels of a pixel.
template<typename T, int SIZE>
struct vec_t {
    T v[SIZE];
};

// A sum to an activation: ReLU, round off ROUND_BITS bits, saturate at the int16 maximum.
template<int ROUND_BITS>
act_t requant_relu(sum_t sum) {
    #pragma HLS INLINE
    if (sum <= 0) return 0;
    const sum_t rounded = (sum + ((sum_t)1 << (ROUND_BITS - 1))) >> ROUND_BITS;
    return rounded > 32767 ? (act_t)32767 : (act_t)rounded;
}

// Round off ROUND_BITS bits, half up.
template<int ROUND_BITS>
sum_t round_shift(sum_t sum) {
    #pragma HLS INLINE
    return ROUND_BITS == 0 ? sum : (sum + ((sum_t)1 << (ROUND_BITS > 0 ? ROUND_BITS - 1 : 0))) >> ROUND_BITS;
}

// A sum to a signed activation, no ReLU: round off ROUND_BITS bits, saturate to int16.
template<int ROUND_BITS>
act_t requant_signed(sum_t sum) {
    #pragma HLS INLINE
    const sum_t rounded = round_shift<ROUND_BITS>(sum);
    return rounded > 32767 ? (act_t)32767 : rounded < -32768 ? (act_t)-32768 : (act_t)rounded;
}

// LeakyReLU with slope LEAK_NUM * 2^-LEAK_FRAC: 41 / 4096 = 0.0100 by default,
// and 41 = 32 + 8 + 1 costs two additions, no multiplier.
template<int LEAK_NUM = 41, int LEAK_FRAC = 12>
sum_t leaky_relu(sum_t value) {
    #pragma HLS INLINE
    if (value >= 0) return value;
    const int64_t scaled = (int64_t)value * LEAK_NUM + ((int64_t)1 << (LEAK_FRAC - 1));
    return (sum_t)(scaled >> LEAK_FRAC);
}

// A sum to a signed activation with LeakyReLU: round, leak, saturate to int16.
template<int ROUND_BITS, int LEAK_NUM = 41, int LEAK_FRAC = 12>
act_t requant_leaky_relu(sum_t sum) {
    #pragma HLS INLINE
    const sum_t leaked = leaky_relu<LEAK_NUM, LEAK_FRAC>(round_shift<ROUND_BITS>(sum));
    return leaked > 32767 ? (act_t)32767 : leaked < -32768 ? (act_t)-32768 : (act_t)leaked;
}

// What a conv sends on: its sums as they are, or with RELU ReLU'd and rounded
// off ROUND_BITS bits to activations.
template<bool RELU, int ROUND_BITS> struct conv_out {
    typedef act_t type;
    static type of(sum_t sum) {
        #pragma HLS INLINE
        return requant_relu<ROUND_BITS>(sum);
    }
};
template<int ROUND_BITS> struct conv_out<false, ROUND_BITS> {
    typedef sum_t type;
    static type of(sum_t sum) {
        #pragma HLS INLINE
        return sum;
    }
};

// A ReLU module's input to an activation: sums are ReLU'd and rounded off ROUND_BITS
// bits; activations (already ReLU'd by their conv) pass as they are.
template<int ROUND_BITS>
act_t to_act(sum_t sum) {
    #pragma HLS INLINE
    return requant_relu<ROUND_BITS>(sum);
}
template<int ROUND_BITS>
act_t to_act(act_t activation) {
    #pragma HLS INLINE
    return activation;
}

static inline act_t max2(act_t a, act_t b) {
    #pragma HLS INLINE
    return a > b ? a : b;
}

// a * b on a DSP or in LUTs ("fabric"), chosen here instead of by HLS. BIND_OP
// takes only a literal word, so each choice is its own branch.
template<bool DSP, typename R, typename A, typename B>
R mul(A a, B b) {
    #pragma HLS INLINE
    R product;
    if (DSP) {
        const R d = (R)a * (R)b;
        #pragma HLS BIND_OP variable=d op=mul impl=dsp
        product = d;
    } else {
        const R f = (R)a * (R)b;
        #pragma HLS BIND_OP variable=f op=mul impl=fabric
        product = f;
    }
    return product;
}
