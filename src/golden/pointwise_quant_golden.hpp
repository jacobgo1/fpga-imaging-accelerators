#pragma once

// A 1x1 conv on one pixel's channels, on quantized weights (scales after the sum).
// The weights keep Conv2d's layout, [out][in][1][1].
template<typename act_t, typename weight_t, int IN_CHANNELS, int OUT_CHANNELS, bool RELU>
void pointwise_quant_golden(const act_t din[IN_CHANNELS],
                            const weight_t weight[OUT_CHANNELS][IN_CHANNELS][1][1], act_t weight_scale,
                            const weight_t bias[OUT_CHANNELS], act_t bias_scale,
                            act_t dout[OUT_CHANNELS]) {
    for (int out = 0; out < OUT_CHANNELS; out++) {
        act_t sum = 0;
        for (int in = 0; in < IN_CHANNELS; in++)
            sum += din[in] * (act_t)weight[out][in][0][0];
        const act_t value = sum * weight_scale + (act_t)bias[out] * bias_scale;
        dout[out] = (RELU && value < 0) ? (act_t)0 : value;
    }
}
