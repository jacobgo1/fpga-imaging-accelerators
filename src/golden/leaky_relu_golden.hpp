#pragma once

// LeakyReLU on a 3D feature map [H][W][CH]: x > 0 ? x : slope * x.
// PyTorch's nn.LeakyReLU(negative_slope=0.01) is the default in WaveletCNN.
template<typename data_t, int H, int W, int CH>
void leaky_relu_golden(const data_t din[H][W][CH], data_t dout[H][W][CH],
                       data_t slope = (data_t)0.01f) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < CH; c++) {
                data_t x = din[i][j][c];
                dout[i][j][c] = x > (data_t)0 ? x : slope * x;
            }
        }
    }
}
