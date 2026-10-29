#pragma once

// dense_golden.hpp on quantized weights, the scales applied after the sum.
template<typename act_t, typename weight_t, int INPUTS, int OUTPUTS>
void dense_quant_golden(const act_t din[INPUTS],
                        const weight_t weight[OUTPUTS][INPUTS], act_t weight_scale,
                        const weight_t bias[OUTPUTS], act_t bias_scale,
                        act_t dout[OUTPUTS]) {
    for (int out = 0; out < OUTPUTS; out++) {
        act_t sum = 0;
        for (int in = 0; in < INPUTS; in++)
            sum += din[in] * (act_t)weight[out][in];
        dout[out] = sum * weight_scale + (act_t)bias[out] * bias_scale;
    }
}
