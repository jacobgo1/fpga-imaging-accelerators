#pragma once

// 2 x 2 max pooling, stride 2.
template<typename data_t, int HEIGHT, int WIDTH, int CHANNELS>
void maxpool_golden(const data_t din[HEIGHT][WIDTH][CHANNELS], data_t dout[HEIGHT / 2][WIDTH / 2][CHANNELS]) {
    for (int row = 0; row < HEIGHT / 2; row++)
        for (int col = 0; col < WIDTH / 2; col++)
            for (int ch = 0; ch < CHANNELS; ch++) {
                data_t largest = din[2 * row][2 * col][ch];
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                        if (din[2 * row + dy][2 * col + dx][ch] > largest)
                            largest = din[2 * row + dy][2 * col + dx][ch];
                dout[row][col][ch] = largest;
            }
}
