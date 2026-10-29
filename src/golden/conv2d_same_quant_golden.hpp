#pragma once

// conv2d_same_golden.hpp on quantized weights: the sum multiplies by q, the
// scales (2^-frac) are applied once, after it. weight_t is int8_t, or float
// holding the same integers. RELU applies a ReLU as each output is written.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS,
         int KERNEL, bool RELU = false>
void conv2d_same_quant_golden(const act_t din[HEIGHT][WIDTH][IN_CHANNELS],
                              const weight_t weight[OUT_CHANNELS][IN_CHANNELS][KERNEL][KERNEL], act_t weight_scale,
                              const weight_t bias[OUT_CHANNELS], act_t bias_scale,
                              act_t dout[HEIGHT][WIDTH][OUT_CHANNELS]) {
    constexpr int PAD = (KERNEL - 1) / 2;
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int out = 0; out < OUT_CHANNELS; out++) {
                act_t sum = 0;
                for (int in = 0; in < IN_CHANNELS; in++)
                    for (int ky = 0; ky < KERNEL; ky++)
                        for (int kx = 0; kx < KERNEL; kx++) {
                            const int y = row + ky - PAD, x = col + kx - PAD;
                            if (y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH)
                                sum += din[y][x][in] * (act_t)weight[out][in][ky][kx];
                        }
                const act_t value = sum * weight_scale + (act_t)bias[out] * bias_scale;
                dout[row][col][out] = (RELU && value < 0) ? (act_t)0 : value;
            }
}
