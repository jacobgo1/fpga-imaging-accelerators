#pragma once

// Fully connected layer, PyTorch's nn.Linear(IN, OUT): dout = weight * din +
// bias, weights in PyTorch's native [out][in] layout.
template<typename data_t, int IN, int OUT>
void dense_golden(const data_t din[IN],
                  const data_t weight[OUT][IN],
                  const data_t bias[OUT],
                  data_t dout[OUT]) {
    for (int o = 0; o < OUT; o++) {
        data_t sum = bias[o];
        for (int f = 0; f < IN; f++) {
            sum += din[f] * weight[o][f];
        }
        dout[o] = sum;
    }
}
