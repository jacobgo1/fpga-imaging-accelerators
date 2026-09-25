#include "model.hpp"
#include "weights.hpp"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

// ============================================================================
// Golden Software Reference Implementations
// ============================================================================
static void golden_conv2d(
    const std::vector<data_t>& in,
    int in_h, int in_w, int in_ch,
    int out_ch, int k_size, int stride,
    const data_t* weights,
    const data_t* bias,
    std::vector<data_t>& out,
    bool apply_relu
) {
    int out_h = (in_h - k_size) / stride + 1;
    int out_w = (in_w - k_size) / stride + 1;
    out.resize(out_h * out_w * out_ch);

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int ih_base = oh * stride;
            int iw_base = ow * stride;

            for (int oc = 0; oc < out_ch; ++oc) {
                acc_t acc = (bias != nullptr) ? bias[oc] : 0;

                for (int ic = 0; ic < in_ch; ++ic) {
                    for (int kr = 0; kr < k_size; ++kr) {
                        for (int kc = 0; kc < k_size; ++kc) {
                            int in_idx = ((ih_base + kr) * in_w + (iw_base + kc)) * in_ch + ic;
                            int w_idx = ((oc * in_ch + ic) * k_size + kr) * k_size + kc;
                            acc += static_cast<acc_t>(in[in_idx]) * static_cast<acc_t>(weights[w_idx]);
                        }
                    }
                }

                if (apply_relu && acc < 0) {
                    acc = 0;
                }

                int out_idx = (oh * out_w + ow) * out_ch + oc;
                out[out_idx] = static_cast<data_t>(acc);
            }
        }
    }
}

static void golden_maxpool(
    const std::vector<data_t>& in,
    int in_h, int in_w, int channels,
    int pool_size, int stride,
    std::vector<data_t>& out
) {
    int out_h = in_h / stride;
    int out_w = in_w / stride;
    out.resize(out_h * out_w * channels);

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int ih_base = oh * stride;
            int iw_base = ow * stride;

            for (int c = 0; c < channels; ++c) {
                data_t max_val = in[(ih_base * in_w + iw_base) * channels + c];

                for (int kr = 0; kr < pool_size; ++kr) {
                    for (int kc = 0; kc < pool_size; ++kc) {
                        int idx = ((ih_base + kr) * in_w + (iw_base + kc)) * channels + c;
                        if (in[idx] > max_val) {
                            max_val = in[idx];
                        }
                    }
                }

                int out_idx = (oh * out_w + ow) * channels + c;
                out[out_idx] = max_val;
            }
        }
    }
}

static void golden_dense(
    const std::vector<data_t>& in,
    int in_features, int out_features,
    const data_t* weights,
    const data_t* bias,
    std::vector<result_t>& out
) {
    out.resize(out_features);
    for (int o = 0; o < out_features; ++o) {
        acc_t acc = (bias != nullptr) ? bias[o] : 0;
        for (int i = 0; i < in_features; ++i) {
            acc += static_cast<acc_t>(in[i]) * static_cast<acc_t>(weights[o * in_features + i]);
        }
        out[o] = static_cast<result_t>(acc);
    }
}

int main() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-10, 10);

    const int total_input_samples = model_cfg::IN_H * model_cfg::IN_W * model_cfg::IN_CH;
    std::vector<data_t> input_data(total_input_samples);

    for (int i = 0; i < total_input_samples; ++i) {
        input_data[i] = static_cast<data_t>(dist(rng));
    }

    // 1. Run Golden Software Reference Pipeline
    std::vector<data_t> gold_conv1;
    golden_conv2d(
        input_data,
        model_cfg::IN_H, model_cfg::IN_W, model_cfg::IN_CH,
        model_cfg::L1_OUT_CH, model_cfg::L1_K_SIZE, 1,
        &model_weights::conv1_weights[0][0][0][0],
        model_weights::conv1_bias,
        gold_conv1,
        true
    );

    std::vector<data_t> gold_pool1;
    golden_maxpool(
        gold_conv1,
        model_cfg::L1_OUT_H, model_cfg::L1_OUT_W, model_cfg::L1_OUT_CH,
        model_cfg::POOL_SIZE, model_cfg::POOL_STRIDE,
        gold_pool1
    );

    std::vector<data_t> gold_conv2;
    golden_conv2d(
        gold_pool1,
        model_cfg::POOL_OUT_H, model_cfg::POOL_OUT_W, model_cfg::L1_OUT_CH,
        model_cfg::L2_OUT_CH, model_cfg::L2_K_SIZE, 1,
        &model_weights::conv2_weights[0][0][0][0],
        model_weights::conv2_bias,
        gold_conv2,
        true
    );

    std::vector<result_t> gold_output;
    golden_dense(
        gold_conv2,
        model_cfg::DENSE_IN_FEATURES, model_cfg::DENSE_OUT_FEATURES,
        &model_weights::dense_weights[0][0],
        model_weights::dense_bias,
        gold_output
    );

    // 2. Stream input into hardware HLS model
    hls::stream<data_t> in_stream("tb_in_stream");
    hls::stream<result_t> out_stream("tb_out_stream");

    for (int i = 0; i < total_input_samples; ++i) {
        in_stream.write(input_data[i]);
    }

    // Execute HLS Top Function
    model(in_stream, out_stream);

    // 3. Verify Hardware Results against Golden Reference
    int errors = 0;
    for (int o = 0; o < model_cfg::DENSE_OUT_FEATURES; ++o) {
        result_t hw_val = out_stream.read();
        result_t ref_val = gold_output[o];

        if (hw_val != ref_val) {
            printf("Error at class [%d]: hardware=%d, reference=%d\n", o, hw_val, ref_val);
            errors++;
        } else {
            printf("Class [%d]: score = %6d (PASS)\n", o, hw_val);
        }
    }

    if (errors == 0) {
        printf("\n>>> PASS: End-to-end dataflow model verification successful! <<<\n");
        return 0;
    } else {
        printf("\n>>> FAIL: %d output mismatches detected! <<<\n", errors);
        return 1;
    }
}
