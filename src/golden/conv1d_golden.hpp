#pragma once

// Conv1d without padding along a [position][channel] spectrum: LENGTH positions
// in, LENGTH - KERNEL + 1 out. Weights in PyTorch's layout, [out][in][k].
template<typename data_t, int LENGTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL>
void conv1d_golden(const data_t din[LENGTH][IN_CHANNELS],
                   const data_t weight[OUT_CHANNELS][IN_CHANNELS][KERNEL],
                   const data_t bias[OUT_CHANNELS],
                   data_t dout[LENGTH - KERNEL + 1][OUT_CHANNELS]) {
    for (int pos = 0; pos < LENGTH - KERNEL + 1; pos++)
        for (int out = 0; out < OUT_CHANNELS; out++) {
            data_t sum = bias[out];
            for (int in = 0; in < IN_CHANNELS; in++)
                for (int k = 0; k < KERNEL; k++)
                    sum += din[pos + k][in] * weight[out][in][k];
            dout[pos][out] = sum;
        }
}
