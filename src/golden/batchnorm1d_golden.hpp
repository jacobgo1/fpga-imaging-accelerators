#pragma once

#include <cmath>

// PyTorch's BatchNorm1d in inference mode over a [position][channel] spectrum
// (see conv1d_golden.hpp): per channel, (x - running_mean) / sqrt(running_var +
// eps) * weight + bias, with the stored running statistics. weight is gamma,
// bias is beta, all length CH as in the state_dict. eps is an argument
// (PyTorch's default is 1e-5). The quantized networks have no BatchNorm: it is
// folded into the conv before it (tools/quantize_weights.py); this module is
// for the float reference of the trained model.
template<typename data_t, int L, int CH>
void batchnorm1d_golden(const data_t din[L][CH],
                        const data_t weight[CH], const data_t bias[CH],
                        const data_t running_mean[CH], const data_t running_var[CH],
                        data_t eps,
                        data_t dout[L][CH]) {
    for (int j = 0; j < L; j++) {
        for (int c = 0; c < CH; c++) {
            const data_t normalized = (din[j][c] - running_mean[c]) / std::sqrt(running_var[c] + eps);
            dout[j][c] = weight[c] * normalized + bias[c];
        }
    }
}
