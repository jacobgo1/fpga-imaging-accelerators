#pragma once

template<typename data_t, int HEIGHT, int WIDTH, int CHANNELS>
void relu_golden(const data_t din[HEIGHT][WIDTH][CHANNELS], data_t dout[HEIGHT][WIDTH][CHANNELS]) {
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int ch = 0; ch < CHANNELS; ch++)
                dout[row][col][ch] = din[row][col][ch] > 0 ? din[row][col][ch] : 0;
}
