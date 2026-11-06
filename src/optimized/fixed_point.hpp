#pragma once

#include <stdint.h>
#include "../common/hls_stream_shim.hpp"

// Integer fixed point, shared by every module. An activation a means a * 2^-F,
// a weight q means q * 2^-frac, so products and their sums are exact in int32.

typedef int16_t act_t;   // an activation
typedef int8_t  act8_t;  // an int8 activation, -127 .. 127
typedef uint8_t uact_t;  // a uint8 activation after ReLU, 0 .. 255 (the 1D networks)
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

// The same to a uint8 activation, saturated at ACT_MAX.
template<int ROUND_BITS, int ACT_MAX>
uact_t requant_relu8(sum_t sum) {
    #pragma HLS INLINE
    if (sum <= 0) return 0;
    const sum_t rounded = (sum + ((sum_t)1 << (ROUND_BITS - 1))) >> ROUND_BITS;
    return rounded > ACT_MAX ? (uact_t)ACT_MAX : (uact_t)rounded;
}

// Round off ROUND_BITS bits, half up. A negative ROUND_BITS gives that many bits more,
// exactly (a shift left): sp_unet_SIZE_opt_u8's last sums have fewer fraction bits than
// the scores.
template<int ROUND_BITS>
sum_t round_shift(sum_t sum) {
    #pragma HLS INLINE
    return ROUND_BITS == 0 ? sum
         : ROUND_BITS < 0 ? (sum_t)((uint32_t)sum << (ROUND_BITS < 0 ? -ROUND_BITS : 0))
         : (sum + ((sum_t)1 << (ROUND_BITS > 0 ? ROUND_BITS - 1 : 0))) >> (ROUND_BITS > 0 ? ROUND_BITS : 0);
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

// The activation after a layer, as in the PyTorch model, inside the layer's own
// process: none sends the sums on (the scores); relu and leaky round off
// ROUND_BITS bits to int16 activations, relu8 to uint8 ones (the 1D networks);
// leaky_scores rounds and leaks but keeps 32 bits, for a head whose scores are
// LeakyReLU'd.
enum class activation { none, relu, relu8, leaky, leaky_scores };

template<activation A, int ROUND_BITS> struct conv_out {
    typedef act_t type;
    static type of(sum_t sum) {
        #pragma HLS INLINE
        return requant_relu<ROUND_BITS>(sum);
    }
};
template<int ROUND_BITS> struct conv_out<activation::none, ROUND_BITS> {
    typedef sum_t type;
    static type of(sum_t sum) {
        #pragma HLS INLINE
        return sum;
    }
};
template<int ROUND_BITS> struct conv_out<activation::relu8, ROUND_BITS> {
    typedef uact_t type;
    static type of(sum_t sum) {
        #pragma HLS INLINE
        return requant_relu8<ROUND_BITS, 255>(sum);
    }
};
template<int ROUND_BITS> struct conv_out<activation::leaky, ROUND_BITS> {
    typedef act_t type;
    static type of(sum_t sum) {
        #pragma HLS INLINE
        return requant_leaky_relu<ROUND_BITS>(sum);
    }
};
template<int ROUND_BITS> struct conv_out<activation::leaky_scores, ROUND_BITS> {
    typedef sum_t type;
    static type of(sum_t sum) {
        #pragma HLS INLINE
        return leaky_relu<>(round_shift<ROUND_BITS>(sum));
    }
};

template<typename T>
T max2(T a, T b) {
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
