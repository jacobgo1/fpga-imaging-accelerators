#include "model.hpp"
#include "weights.hpp"
#include "layers_conv2d.hpp"
#include "layers_pool.hpp"
#include "layers_dense.hpp"

// ============================================================================
// Top-Level Model Accelerator
//
// Chains multi-layer CNN components concurrently using #pragma HLS DATAFLOW.
// Features stream through inter-layer FIFOs without any whole-image BRAM buffering.
// ============================================================================
void model(
    hls::stream<data_t>& in_stream,
    hls::stream<result_t>& out_stream
) {
    #pragma HLS INTERFACE mode=axis port=in_stream
    #pragma HLS INTERFACE mode=axis port=out_stream
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    // Inter-layer FIFOs
    hls::stream<data_t> s_conv1("s_conv1");
    #pragma HLS STREAM variable=s_conv1 depth=16

    hls::stream<data_t> s_pool1("s_pool1");
    #pragma HLS STREAM variable=s_pool1 depth=16

    hls::stream<data_t> s_conv2("s_conv2");
    #pragma HLS STREAM variable=s_conv2 depth=16

    // Layer 1: Conv2D (16x16x4 -> 14x14x8) + fused ReLU
    conv2d_layer<data_t, data_t, acc_t, data_t,
                 model_cfg::IN_H, model_cfg::IN_W, model_cfg::IN_CH,
                 model_cfg::L1_OUT_CH, model_cfg::L1_K_SIZE, 1, true>(
        in_stream,
        model_weights::conv1_weights,
        model_weights::conv1_bias,
        s_conv1
    );

    // Layer 2: MaxPool2D 2x2 (14x14x8 -> 7x7x8)
    maxpool2d_layer<data_t,
                    model_cfg::L1_OUT_H, model_cfg::L1_OUT_W, model_cfg::L1_OUT_CH,
                    model_cfg::POOL_SIZE, model_cfg::POOL_STRIDE>(
        s_conv1,
        s_pool1
    );

    // Layer 3: Conv2D (7x7x8 -> 5x5x16) + fused ReLU
    conv2d_layer<data_t, data_t, acc_t, data_t,
                 model_cfg::POOL_OUT_H, model_cfg::POOL_OUT_W, model_cfg::L1_OUT_CH,
                 model_cfg::L2_OUT_CH, model_cfg::L2_K_SIZE, 1, true>(
        s_pool1,
        model_weights::conv2_weights,
        model_weights::conv2_bias,
        s_conv2
    );

    // Layer 4: Dense Classifier (400 -> 4)
    dense_layer<data_t, data_t, acc_t, result_t,
                model_cfg::DENSE_IN_FEATURES, model_cfg::DENSE_OUT_FEATURES, false>(
        s_conv2,
        model_weights::dense_weights,
        model_weights::dense_bias,
        out_stream
    );
}
