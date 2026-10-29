#pragma once

#include <cmath>

// BatchNorm1d in inference mode, on a [position][channel] spectrum.
template<typename data_t, int LENGTH, int CHANNELS>
void batchnorm1d_golden(const data_t din[LENGTH][CHANNELS],
                        const data_t weight[CHANNELS], const data_t bias[CHANNELS],
                        const data_t running_mean[CHANNELS], const data_t running_var[CHANNELS],
                        data_t eps,
                        data_t dout[LENGTH][CHANNELS]) {
    for (int pos = 0; pos < LENGTH; pos++)
        for (int ch = 0; ch < CHANNELS; ch++) {
            const data_t normalized = (din[pos][ch] - running_mean[ch]) / std::sqrt(running_var[ch] + eps);
            dout[pos][ch] = weight[ch] * normalized + bias[ch];
        }
}
