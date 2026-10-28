#pragma once

#include "quant_scale.hpp"

// The quantized 1D-SpectralPointwise, for one pixel: its spectrum in, 3 class
// scores out. Three 1x1 convolutions, so each one mixes the channels of the
// pixel and nothing else: a dense layer on the pixel's channels.
//
//   convsp1: 1x1, BANDS -> 24, bias   (BatchNorm folded in) -> ReLU
//   convsp2: 1x1, 24 -> 12, bias                            -> ReLU
//   convsp3: 1x1, 12 -> CLASSES, bias                       -> the scores
//
// The trained model has a BatchNorm after convsp1 (convsp1 -> bnsp1 -> ReLU,
// hypso-onboard-segmentation/model_definition/SpectralPointwise1D.py);
// tools/quantize_weights.py folds it into convsp1's weights and bias, which
// leaves exactly this network.
//
// Weights are the quantized integers q with one power-of-two scale per tensor,
// real value = q * 2^-frac (weights/quantized/spectralpointwise1d/), as in
// justoliunet_golden.hpp: the inner loop multiplies by q, and the scale is
// applied once per output, after the sum. w_t can be int8_t, or float holding
// the same integers (what the HLS kernel uses). The weights keep Conv2d's
// layout, [out][in][1][1], so the header's arrays go in as they are.

// The trained model's sizes.
constexpr int SPW_BANDS = 110;     // bands the model uses: raw 8-117, z-scored
constexpr int SPW_CH1 = 24;        // channels after convsp1
constexpr int SPW_CH2 = 12;        // channels after convsp2
constexpr int SPW_CLASSES = 3;     // 0 cloud, 1 land, 2 sea

// A 1x1 convolution on one pixel, on quantized weights, ReLU or not.
template<typename act_t, typename w_t, int IN, int OUT, bool RELU>
void pointwise_quant_golden(const act_t din[IN],
                            const w_t weight[OUT][IN][1][1], act_t weight_scale,
                            const w_t bias[OUT], act_t bias_scale,
                            act_t dout[OUT]) {
    for (int o = 0; o < OUT; o++) {
        act_t sum = 0;
        for (int i = 0; i < IN; i++) {
            sum += din[i] * (act_t)weight[o][i][0][0];
        }
        const act_t y = sum * weight_scale + (act_t)bias[o] * bias_scale;
        dout[o] = RELU && y < 0 ? (act_t)0 : y;
    }
}

template<typename act_t, typename w_t, int BANDS, int CH1, int CH2, int CLASSES>
void spectralpointwise1d_golden(
    const act_t din[BANDS],
    const w_t w1[CH1][BANDS][1][1], act_t w1_scale, const w_t b1[CH1],     act_t b1_scale,
    const w_t w2[CH2][CH1][1][1],   act_t w2_scale, const w_t b2[CH2],     act_t b2_scale,
    const w_t w3[CLASSES][CH2][1][1], act_t w3_scale, const w_t b3[CLASSES], act_t b3_scale,
    act_t dout[CLASSES])
{
    act_t h1[CH1], h2[CH2];
    pointwise_quant_golden<act_t, w_t, BANDS, CH1, true>(din, w1, w1_scale, b1, b1_scale, h1);
    pointwise_quant_golden<act_t, w_t, CH1, CH2, true>(h1, w2, w2_scale, b2, b2_scale, h2);
    // The head: no activation (the scores).
    pointwise_quant_golden<act_t, w_t, CH2, CLASSES, false>(h2, w3, w3_scale, b3, b3_scale, dout);
}
