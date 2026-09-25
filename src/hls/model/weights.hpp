#ifndef MODEL_WEIGHTS_HPP
#define MODEL_WEIGHTS_HPP

#include "model.hpp"

// ============================================================================
// On-Chip Model Weights & Biases
//
// Stored as static const arrays initialized directly in on-chip Block RAM / ROM.
// This completely avoids external memory bus contention and DRAM latency during
// inference.
//
// In production, these arrays are exported directly from trained PyTorch /
// TensorFlow quantization pipelines.
// ============================================================================
namespace model_weights {

// Conv1 Weights: [OUT_CH=8][IN_CH=4][KH=3][KW=3] = 288 params
static const data_t conv1_weights[model_cfg::L1_OUT_CH][model_cfg::IN_CH]
                                 [model_cfg::L1_K_SIZE][model_cfg::L1_K_SIZE] = {
    #include "conv1_weights.inc"
};

static const data_t conv1_bias[model_cfg::L1_OUT_CH] = {
    1, 0, -1, 2, 0, 1, -2, 1
};

// Conv2 Weights: [OUT_CH=16][IN_CH=8][KH=3][KW=3] = 1152 params
static const data_t conv2_weights[model_cfg::L2_OUT_CH][model_cfg::L1_OUT_CH]
                                 [model_cfg::L2_K_SIZE][model_cfg::L2_K_SIZE] = {
    #include "conv2_weights.inc"
};

static const data_t conv2_bias[model_cfg::L2_OUT_CH] = {
    0, 1, 0, -1, 2, 0, 1, -1, 0, 1, -1, 0, 1, 2, 0, -1
};

// Dense Weights: [OUT=4][IN=400] = 1600 params
static const data_t dense_weights[model_cfg::DENSE_OUT_FEATURES]
                                 [model_cfg::DENSE_IN_FEATURES] = {
    #include "dense_weights.inc"
};

static const data_t dense_bias[model_cfg::DENSE_OUT_FEATURES] = {
    2, -1, 1, 0
};

} // namespace model_weights

#endif // MODEL_WEIGHTS_HPP
