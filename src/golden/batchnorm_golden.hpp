#pragma once

#include <cmath>

// BatchNorm2d in inference mode: the running statistics, not the batch's.
template<typename data_t, int HEIGHT, int WIDTH, int CHANNELS>
void batchnorm_golden(const data_t din[HEIGHT][WIDTH][CHANNELS],
                      const data_t weight[CHANNELS], const data_t bias[CHANNELS],
                      const data_t running_mean[CHANNELS], const data_t running_var[CHANNELS],
                      data_t eps,
                      data_t dout[HEIGHT][WIDTH][CHANNELS]) {
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int ch = 0; ch < CHANNELS; ch++) {
                const data_t normalized = (din[row][col][ch] - running_mean[ch]) / std::sqrt(running_var[ch] + eps);
                dout[row][col][ch] = weight[ch] * normalized + bias[ch];
            }
}
