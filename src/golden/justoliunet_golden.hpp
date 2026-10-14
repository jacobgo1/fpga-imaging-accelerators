#pragma once

#include "quant_scale.hpp"
#include "relu1d_golden.hpp"
#include "maxpool1d_golden.hpp"
#include "flatten_golden.hpp"

// The quantized 1D-Justo-LiuNet, for one pixel: its spectrum in, 3 class
// scores out. A 1D network: every convolution runs along the bands of the
// pixel (spectrally), never across pixels, so pixels are classified
// independently.
//
//   block 1: conv1d(1 -> BASE_CH,      k = K) -> ReLU -> maxpool(2)
//   block 2: conv1d(BASE_CH -> 2 BASE_CH)     -> ReLU -> maxpool(2)
//   block 3: conv1d(2 BASE_CH -> 3 BASE_CH)   -> ReLU -> maxpool(2)
//   block 4: conv1d(3 BASE_CH -> 4 BASE_CH)   -> ReLU -> maxpool(2)
//   flatten (channel-major, as torch.flatten) -> dense(-> CLASSES)
//
// The convolutions are valid (no padding), the pools drop an odd last
// position: with 110 bands, K = 6 and BASE_CH = 6 the spectrum goes
// 110 -> 105 -> 52 -> 47 -> 23 -> 18 -> 9 -> 4 -> 2, and the dense layer gets
// 24 channels x 2 = 48 values, as the checkpoints' fc.weight [3, 48] says.
//
// The same network serves both trained models. justoliunet_bn has a BatchNorm
// after conv1 (conv1 -> bn1 -> ReLU: on the labelled aegean pixels that order
// gives 96.7 % accuracy, any later place 63 %); tools/quantize_weights.py folds
// it into conv1's weights and bias, which leaves exactly this network.
//
// Weights are the quantized integers q with one power-of-two scale per tensor,
// real value = q * 2^-frac (weights/quantized/<model>/<model>_int8.hpp), as in
// justounetsimple_golden.hpp: the inner loop multiplies by q, and the scale is
// applied once per output, after the sum. w_t can be int8_t, or float holding
// the same integers (what the HLS kernels use). Activations are
// [position][channel] (see conv1d_golden.hpp).

// The trained models' sizes.
constexpr int JLN_BANDS = 110;     // bands the models use: raw 8-117, z-scored
constexpr int JLN_K = 6;           // kernel size of every conv
constexpr int JLN_BASE_CH = 6;     // channels of conv1; conv2-4 have 2x, 3x, 4x
constexpr int JLN_CLASSES = 3;     // 0 cloud, 1 land, 2 sea

// The spectrum's length after one block (valid conv, then pool by 2), and after
// all four: what flatten hands the dense layer, per channel.
constexpr int justoliunet_block_len(int len, int K) { return (len - K + 1) / 2; }
constexpr int justoliunet_flat_len(int len, int K) {
    return justoliunet_block_len(justoliunet_block_len(justoliunet_block_len(
               justoliunet_block_len(len, K), K), K), K);
}

// Valid 1D convolution on quantized weights (conv1d_golden.hpp, with the
// scales applied after the sum).
template<typename act_t, typename w_t, int L, int IN_CH, int OUT_CH, int K>
void conv1d_quant_golden(const act_t din[L][IN_CH],
                         const w_t weight[OUT_CH][IN_CH][K], act_t weight_scale,
                         const w_t bias[OUT_CH], act_t bias_scale,
                         act_t dout[L - K + 1][OUT_CH]) {
    static_assert(L >= K, "the spectrum must be at least as long as the kernel");
    for (int j = 0; j < L - K + 1; j++) {
        for (int oc = 0; oc < OUT_CH; oc++) {
            act_t sum = 0;
            for (int ic = 0; ic < IN_CH; ic++) {
                for (int k = 0; k < K; k++) {
                    sum += din[j + k][ic] * (act_t)weight[oc][ic][k];
                }
            }
            dout[j][oc] = sum * weight_scale + (act_t)bias[oc] * bias_scale;
        }
    }
}

// Dense layer on quantized weights (dense_golden.hpp, with the scales).
template<typename act_t, typename w_t, int IN, int OUT>
void dense_quant_golden(const act_t din[IN],
                        const w_t weight[OUT][IN], act_t weight_scale,
                        const w_t bias[OUT], act_t bias_scale,
                        act_t dout[OUT]) {
    for (int o = 0; o < OUT; o++) {
        act_t sum = 0;
        for (int f = 0; f < IN; f++) {
            sum += din[f] * (act_t)weight[o][f];
        }
        dout[o] = sum * weight_scale + (act_t)bias[o] * bias_scale;
    }
}

template<typename act_t, typename w_t, int L, int BASE_CH, int CLASSES, int K>
void justoliunet_golden(
    const act_t din[L][1],
    const w_t w1[BASE_CH][1][K],             act_t w1_scale, const w_t b1[BASE_CH],   act_t b1_scale,
    const w_t w2[2*BASE_CH][BASE_CH][K],     act_t w2_scale, const w_t b2[2*BASE_CH], act_t b2_scale,
    const w_t w3[3*BASE_CH][2*BASE_CH][K],   act_t w3_scale, const w_t b3[3*BASE_CH], act_t b3_scale,
    const w_t w4[4*BASE_CH][3*BASE_CH][K],   act_t w4_scale, const w_t b4[4*BASE_CH], act_t b4_scale,
    const w_t wf[CLASSES][4*BASE_CH * justoliunet_flat_len(L, K)], act_t wf_scale,
    const w_t bf[CLASSES], act_t bf_scale,
    act_t dout[CLASSES])
{
    // Lengths after each conv (C) and each pool (P).
    constexpr int C1 = L  - K + 1, P1 = C1 / 2;
    constexpr int C2 = P1 - K + 1, P2 = C2 / 2;
    constexpr int C3 = P2 - K + 1, P3 = C3 / 2;
    constexpr int C4 = P3 - K + 1, P4 = C4 / 2;
    constexpr int FLAT = 4 * BASE_CH * P4;
    static_assert(P4 >= 1, "the spectrum is too short for four conv + pool blocks");
    static_assert(P4 == justoliunet_flat_len(L, K), "the block lengths must agree");

    // ---- block 1 ----
    act_t conv1[C1][BASE_CH], relu1[C1][BASE_CH], pool1[P1][BASE_CH];
    conv1d_quant_golden<act_t, w_t, L, 1, BASE_CH, K>(din, w1, w1_scale, b1, b1_scale, conv1);
    relu1d_golden<act_t, C1, BASE_CH>(conv1, relu1);
    maxpool1d_golden<act_t, C1, BASE_CH, 2>(relu1, pool1);

    // ---- block 2 ----
    act_t conv2[C2][2*BASE_CH], relu2[C2][2*BASE_CH], pool2[P2][2*BASE_CH];
    conv1d_quant_golden<act_t, w_t, P1, BASE_CH, 2*BASE_CH, K>(pool1, w2, w2_scale, b2, b2_scale, conv2);
    relu1d_golden<act_t, C2, 2*BASE_CH>(conv2, relu2);
    maxpool1d_golden<act_t, C2, 2*BASE_CH, 2>(relu2, pool2);

    // ---- block 3 ----
    act_t conv3[C3][3*BASE_CH], relu3[C3][3*BASE_CH], pool3[P3][3*BASE_CH];
    conv1d_quant_golden<act_t, w_t, P2, 2*BASE_CH, 3*BASE_CH, K>(pool2, w3, w3_scale, b3, b3_scale, conv3);
    relu1d_golden<act_t, C3, 3*BASE_CH>(conv3, relu3);
    maxpool1d_golden<act_t, C3, 3*BASE_CH, 2>(relu3, pool3);

    // ---- block 4 ----
    act_t conv4[C4][4*BASE_CH], relu4[C4][4*BASE_CH], pool4[P4][4*BASE_CH];
    conv1d_quant_golden<act_t, w_t, P3, 3*BASE_CH, 4*BASE_CH, K>(pool3, w4, w4_scale, b4, b4_scale, conv4);
    relu1d_golden<act_t, C4, 4*BASE_CH>(conv4, relu4);
    maxpool1d_golden<act_t, C4, 4*BASE_CH, 2>(relu4, pool4);

    // ---- head: flatten, dense; no activation (the scores) ----
    act_t flat[FLAT];
    flatten_golden<act_t, P4, 4*BASE_CH>(pool4, flat);
    dense_quant_golden<act_t, w_t, FLAT, CLASSES>(flat, wf, wf_scale, bf, bf_scale, dout);
}
