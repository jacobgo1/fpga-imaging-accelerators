#pragma once

// nn.Linear: weights in PyTorch's layout, [out][in].
template<typename data_t, int INPUTS, int OUTPUTS>
void dense_golden(const data_t din[INPUTS],
                  const data_t weight[OUTPUTS][INPUTS],
                  const data_t bias[OUTPUTS],
                  data_t dout[OUTPUTS]) {
    for (int out = 0; out < OUTPUTS; out++) {
        data_t sum = bias[out];
        for (int in = 0; in < INPUTS; in++)
            sum += din[in] * weight[out][in];
        dout[out] = sum;
    }
}
