#pragma once

#include <stdint.h>
#include "../../common/hls_stream_shim.hpp"

// Integer fixed point, shared by every module. An activation a means a * 2^-F,
// a weight q means q * 2^-frac, so products and their sums are exact in int32.

typedef int16_t act_t;   // an activation
typedef int8_t  act8_t;  // an int8 activation, -127 .. 127
typedef int8_t  wgt_t;   // a quantized weight
typedef int32_t acc_t;   // a sum of products, or a bias at the same scale

// SIZE values that travel together through a stream, e.g. the channels of a pixel.
template<typename T, int SIZE>
struct vec_t {
    T v[SIZE];
};

// A sum to an activation: ReLU, round off SHIFT bits, saturate at the int16 maximum.
template<int SHIFT>
act_t requant_relu(acc_t sum) {
    #pragma HLS INLINE
    if (sum <= 0) return 0;
    const acc_t rounded = (sum + ((acc_t)1 << (SHIFT - 1))) >> SHIFT;
    return rounded > 32767 ? (act_t)32767 : (act_t)rounded;
}

// Round off SHIFT bits, half up.
template<int SHIFT>
acc_t round_shift(acc_t sum) {
    #pragma HLS INLINE
    return SHIFT == 0 ? sum : (sum + ((acc_t)1 << (SHIFT > 0 ? SHIFT - 1 : 0))) >> SHIFT;
}

// A sum to a signed activation, no ReLU: round off SHIFT bits, saturate to int16.
template<int SHIFT>
act_t requant_act(acc_t sum) {
    #pragma HLS INLINE
    const acc_t rounded = round_shift<SHIFT>(sum);
    return rounded > 32767 ? (act_t)32767 : rounded < -32768 ? (act_t)-32768 : (act_t)rounded;
}

// LeakyReLU with slope LEAK_NUM * 2^-LEAK_FRAC: 41 / 4096 = 0.0100 by default,
// and 41 = 32 + 8 + 1 costs two additions, no multiplier.
template<int LEAK_NUM = 41, int LEAK_FRAC = 12>
acc_t leaky_relu(acc_t value) {
    #pragma HLS INLINE
    if (value >= 0) return value;
    const int64_t scaled = (int64_t)value * LEAK_NUM + ((int64_t)1 << (LEAK_FRAC - 1));
    return (acc_t)(scaled >> LEAK_FRAC);
}

// A sum to a signed activation with LeakyReLU: round, leak, saturate to int16.
template<int SHIFT, int LEAK_NUM = 41, int LEAK_FRAC = 12>
act_t requant_leaky_relu(acc_t sum) {
    #pragma HLS INLINE
    const acc_t leaked = leaky_relu<LEAK_NUM, LEAK_FRAC>(round_shift<SHIFT>(sum));
    return leaked > 32767 ? (act_t)32767 : leaked < -32768 ? (act_t)-32768 : (act_t)leaked;
}

// What a conv sends on: its sums as they are (RELU_SHIFT 0), or ReLU'd and
// rounded off RELU_SHIFT bits to activations.
template<int RELU_SHIFT> struct sums_out {
    typedef act_t type;
    static type of(acc_t sum) {
        #pragma HLS INLINE
        return requant_relu<RELU_SHIFT>(sum);
    }
};
template<> struct sums_out<0> {
    typedef acc_t type;
    static type of(acc_t sum) {
        #pragma HLS INLINE
        return sum;
    }
};

// A ReLU module's input to an activation: sums are ReLU'd and rounded off SHIFT
// bits; activations (already ReLU'd by their conv) pass as they are.
template<int SHIFT>
act_t relu_in(acc_t sum) {
    #pragma HLS INLINE
    return requant_relu<SHIFT>(sum);
}
template<int SHIFT>
act_t relu_in(act_t activation) {
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
