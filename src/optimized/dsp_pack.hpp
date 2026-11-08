#pragma once

#include <type_traits>
#include "fixed_point.hpp"

// Two multiplies in one DSP (AMD WP486). Output channels 2j and 2j + 1 share the
// activation a, so their weights are packed
//     (w_high * 2^SHIFT + w_low) * a = w_high * a * 2^SHIFT + w_low * a
// and a sum of GROUP such products still holds both sums, as long as the low one
// fits SHIFT signed bits. Exact, bit for bit.
//
// SHIFT is the chip's, set by the DSP's wide input (SHIFT + 9 bits for the packed weight):
//   UltraScale+ DSP48E2 (27 bits): 18;   7-series DSP48E1 (25 bits): 16.
// GROUP follows from SHIFT and the activations' range (pack_group):
//   int8, -127 .. 127 (never -128):  8 on the DSP48E2, 2 on the DSP48E1;
//   uint8, 0 .. 255:                 4 and 1.
//
// A module packs the weights once, at load (pack_weights), adds GROUP packed_mul
// products and splits the sum with unpack_add. Packed next to the multiply instead,
// HLS put the addition in the DSP's pre-adder and the RTL came out wrong (cosim failed).

#if defined(__VITIS_HLS__) || defined(__SYNTHESIS__)
#include <ap_int.h>
template<int SHIFT> using packed_weight_t = ap_int<SHIFT + 9>;
typedef ap_int<48> packed_sum_t;                 // the DSP's accumulator
#else
template<int SHIFT> using packed_weight_t = int64_t;
typedef int64_t packed_sum_t;
#endif

// The most products a group can add: |w * a| <= 128 * A_MAX each, and the low
// sum must stay within SHIFT signed bits.
template<int SHIFT, int A_MAX>
constexpr int pack_group() {
    return ((1 << (SHIFT - 1)) - 1) / (128 * A_MAX);
}
static_assert(pack_group<18, 127>() == 8 && pack_group<16, 127>() == 2, "int8 groups");
static_assert(pack_group<18, 255>() == 4 && pack_group<16, 255>() == 1, "uint8 groups");

// The largest input a packed group must hold: uint8 255, int8 127.
template<typename IN_T> constexpr int input_max() { return std::is_same<IN_T, uact_t>::value ? 255 : 127; }

// Channel 2j + 1's weight above channel 2j's. At load only (see above).
template<int SHIFT>
packed_weight_t<SHIFT> pack_weights(weight_t high, weight_t low) {
    #pragma HLS INLINE
    return (packed_weight_t<SHIFT>)high * (packed_weight_t<SHIFT>)(1 << SHIFT) + (packed_weight_t<SHIFT>)low;
}

// One packed product, on a DSP. a as it is: a uint8 widened first would need a
// 27 x 48 multiplier.
template<int SHIFT, typename A>
packed_sum_t packed_mul(packed_weight_t<SHIFT> w, A a) {
    #pragma HLS INLINE
    const packed_sum_t prod = w * a;
    #pragma HLS BIND_OP variable=prod op=mul impl=dsp
    return prod;
}

// A group's sum to the low channel's part: its bottom SHIFT bits, sign-extended.
template<int SHIFT>
sum_t low_sum(int64_t sum) {
    #pragma HLS INLINE
    return (sum_t)((uint32_t)(uint64_t)sum << (32 - SHIFT)) >> (32 - SHIFT);
}

// ... and the high channel's: the rest, plus the 1 the low part borrowed when negative.
template<int SHIFT>
sum_t high_sum(int64_t sum) {
    #pragma HLS INLINE
    return (sum_t)(sum >> SHIFT) + (sum_t)((sum >> (SHIFT - 1)) & 1);
}

// A group's sum split back into its two channels, added to low and high.
template<int SHIFT>
void unpack_add(packed_sum_t sum, sum_t& low, sum_t& high) {
    #pragma HLS INLINE
    low += low_sum<SHIFT>((int64_t)sum);
    high += high_sum<SHIFT>((int64_t)sum);
}
