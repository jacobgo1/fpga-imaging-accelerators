#pragma once

// Conv2d with zero padding (K - 1) / 2, so the output is the input's size.
// Taps outside the image are skipped. Weights in PyTorch's layout, [out][in][ky][kx].
template<typename data_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL>
void conv2d_same_golden(const data_t din[HEIGHT][WIDTH][IN_CHANNELS],
                        const data_t weight[OUT_CHANNELS][IN_CHANNELS][KERNEL][KERNEL],
                        const data_t bias[OUT_CHANNELS],
                        data_t dout[HEIGHT][WIDTH][OUT_CHANNELS]) {
    constexpr int PAD = (KERNEL - 1) / 2;
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int out = 0; out < OUT_CHANNELS; out++) {
                data_t sum = bias[out];
                for (int in = 0; in < IN_CHANNELS; in++)
                    for (int ky = 0; ky < KERNEL; ky++)
                        for (int kx = 0; kx < KERNEL; kx++) {
                            const int y = row + ky - PAD, x = col + kx - PAD;
                            if (y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH)
                                sum += din[y][x][in] * weight[out][in][ky][kx];
                        }
                dout[row][col][out] = sum;
            }
}
