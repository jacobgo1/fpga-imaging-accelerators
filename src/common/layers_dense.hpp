#ifndef COMMON_LAYERS_DENSE_HPP
#define COMMON_LAYERS_DENSE_HPP

#include "types.hpp"

// ============================================================================
// Generic Streaming Dense (Fully Connected) Layer
//
// Parameters:
//   T_IN         - Input activation type
//   T_WEIGHT     - Weight type
//   T_ACC        - Accumulator type
//   T_OUT        - Output activation type
//   IN_FEATURES  - Number of input features
//   OUT_FEATURES - Number of output classes / features
//   APPLY_RELU   - If true, applies ReLU to the output
// ============================================================================
template <
    typename T_IN,
    typename T_WEIGHT,
    typename T_ACC,
    typename T_OUT,
    int IN_FEATURES,
    int OUT_FEATURES,
    bool APPLY_RELU = false
>
void dense_layer(
    hls::stream<T_IN>& in_stream,
    const T_WEIGHT weights[OUT_FEATURES][IN_FEATURES],
    const T_WEIGHT bias[OUT_FEATURES],
    hls::stream<T_OUT>& out_stream
) {
    #pragma HLS INLINE off

    T_IN input_vec[IN_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=input_vec cyclic factor=8

    // Read full input feature vector
    read_dense_in:
    for (int i = 0; i < IN_FEATURES; ++i) {
        #pragma HLS PIPELINE II=1
        input_vec[i] = in_stream.read();
    }

    // Compute dot product for each output feature
    dense_out_loop:
    for (int o = 0; o < OUT_FEATURES; ++o) {
        #pragma HLS PIPELINE II=1
        T_ACC acc = (bias != nullptr) ? static_cast<T_ACC>(bias[o]) : T_ACC(0);

        dense_mac:
        for (int i = 0; i < IN_FEATURES; ++i) {
            #pragma HLS UNROLL factor=8
            acc += static_cast<T_ACC>(input_vec[i]) * static_cast<T_ACC>(weights[o][i]);
        }

        if (APPLY_RELU && acc < T_ACC(0)) {
            acc = T_ACC(0);
        }

        out_stream.write(static_cast<T_OUT>(acc));
    }
}

#endif // COMMON_LAYERS_DENSE_HPP
