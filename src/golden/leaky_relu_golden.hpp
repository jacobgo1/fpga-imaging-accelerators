#pragma once

template<typename data_t, int HEIGHT, int WIDTH, int CHANNELS>
void leaky_relu_golden(const data_t din[HEIGHT][WIDTH][CHANNELS], data_t dout[HEIGHT][WIDTH][CHANNELS],
                       data_t slope = (data_t)0.01f) {
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int ch = 0; ch < CHANNELS; ch++) {
                const data_t x = din[row][col][ch];
                dout[row][col][ch] = x > (data_t)0 ? x : slope * x;
            }
}
