#ifndef MODEL_HPP
#define MODEL_HPP

#include "types.hpp"

// ============================================================================
// Model Architecture Dimensions
// Multi-layer CNN pipeline with dataflow streaming:
// Input (16x16x4) -> Conv2D (14x14x8) + ReLU -> MaxPool (7x7x8)
//                 -> Conv2D (5x5x16) + ReLU  -> Dense (4 output classes)
// ============================================================================
namespace model_cfg {
    // Input Dimensions
    constexpr int IN_H  = 16;
    constexpr int IN_W  = 16;
    constexpr int IN_CH = 4;

    // Layer 1: Conv2D (16x16x4 -> 14x14x8)
    constexpr int L1_OUT_CH = 8;
    constexpr int L1_K_SIZE = 3;
    constexpr int L1_OUT_H  = IN_H - L1_K_SIZE + 1; // 14
    constexpr int L1_OUT_W  = IN_W - L1_K_SIZE + 1; // 14

    // Layer 2: MaxPool2D 2x2 (14x14x8 -> 7x7x8)
    constexpr int POOL_SIZE   = 2;
    constexpr int POOL_STRIDE = 2;
    constexpr int POOL_OUT_H  = L1_OUT_H / POOL_STRIDE; // 7
    constexpr int POOL_OUT_W  = L1_OUT_W / POOL_STRIDE; // 7

    // Layer 3: Conv2D (7x7x8 -> 5x5x16)
    constexpr int L2_OUT_CH = 16;
    constexpr int L2_K_SIZE = 3;
    constexpr int L2_OUT_H  = POOL_OUT_H - L2_K_SIZE + 1; // 5
    constexpr int L2_OUT_W  = POOL_OUT_W - L2_K_SIZE + 1; // 5

    // Layer 4: Dense Classifier (400 -> 4)
    constexpr int DENSE_IN_FEATURES  = L2_OUT_H * L2_OUT_W * L2_OUT_CH; // 400
    constexpr int DENSE_OUT_FEATURES = 4;
}

// Top-level synthesizable entry point
void model(
    hls::stream<data_t>& in_stream,
    hls::stream<result_t>& out_stream
);

#endif // MODEL_HPP
