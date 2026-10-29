#pragma once

// conv1d_golden.hpp on quantized weights: the sum multiplies by q, the scales
// (2^-frac) are applied once, after it. weight_t is int8_t, or float holding the
// same integers.
template<typename act_t, typename weight_t, int LENGTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL>
void conv1d_quant_golden(const act_t din[LENGTH][IN_CHANNELS],
                         const weight_t weight[OUT_CHANNELS][IN_CHANNELS][KERNEL], act_t weight_scale,
                         const weight_t bias[OUT_CHANNELS], act_t bias_scale,
                         act_t dout[LENGTH - KERNEL + 1][OUT_CHANNELS]) {
    for (int pos = 0; pos < LENGTH - KERNEL + 1; pos++)
        for (int out = 0; out < OUT_CHANNELS; out++) {
            act_t sum = 0;
            for (int in = 0; in < IN_CHANNELS; in++)
                for (int k = 0; k < KERNEL; k++)
                    sum += din[pos + k][in] * (act_t)weight[out][in][k];
            dout[pos][out] = sum * weight_scale + (act_t)bias[out] * bias_scale;
        }
}
