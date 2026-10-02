#ifndef COMMON_LAYERS_ACTIVATION_HPP
#define COMMON_LAYERS_ACTIVATION_HPP

#include "types.hpp"

// ============================================================================
// Streaming ReLU Activation
// Streams in TOTAL_SAMPLES, applies ReLU (max(0, x)), and writes to out_stream
// with II=1 pipelined throughput.
// ============================================================================
template <typename T, int TOTAL_SAMPLES>
void relu_layer(hls::stream<T>& in_stream, hls::stream<T>& out_stream) {
    for (int i = 0; i < TOTAL_SAMPLES; ++i) {
        #pragma HLS PIPELINE II=1
        T val = in_stream.read();
        out_stream.write((val > T(0)) ? val : T(0));
    }
}

// ============================================================================
// Streaming Requantization / Scaling + ReLU Activation
// Scales accumulator down by a power of 2 (arithmetic right shift) or scale factor,
// clamps to target bitwidth range, and applies ReLU.
// ============================================================================
template <typename T_IN, typename T_OUT, int TOTAL_SAMPLES, int SHIFT_BITS = 0>
void requantize_relu_layer(hls::stream<T_IN>& in_stream, hls::stream<T_OUT>& out_stream) {
    for (int i = 0; i < TOTAL_SAMPLES; ++i) {
        #pragma HLS PIPELINE II=1
        T_IN in_val = in_stream.read();
        T_IN shifted = (SHIFT_BITS > 0) ? (in_val >> SHIFT_BITS) : in_val;
        T_IN clamped = (shifted > T_IN(0)) ? shifted : T_IN(0);
        // Saturate to T_OUT range if needed
        out_stream.write(static_cast<T_OUT>(clamped));
    }
}

#endif // COMMON_LAYERS_ACTIVATION_HPP
