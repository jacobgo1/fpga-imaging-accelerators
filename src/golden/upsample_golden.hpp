#pragma once

// Nearest-neighbour upsampling by 2: each pixel becomes a 2 x 2 block.
template<typename data_t, int HEIGHT, int WIDTH, int CHANNELS>
void upsample_golden(const data_t din[HEIGHT][WIDTH][CHANNELS], data_t dout[2 * HEIGHT][2 * WIDTH][CHANNELS]) {
    for (int row = 0; row < 2 * HEIGHT; row++)
        for (int col = 0; col < 2 * WIDTH; col++)
            for (int ch = 0; ch < CHANNELS; ch++)
                dout[row][col][ch] = din[row / 2][col / 2][ch];
}
