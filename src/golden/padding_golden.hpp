#pragma once

template<typename data_t, int H, int W, int CH, int PAD>
void padding_golden(data_t din[H][W][CH], data_t dout[H + 2*PAD][W + 2*PAD][CH]) {
    for (int i = 0; i < H + 2*PAD; i++) {
        for (int j = 0; j < W + 2*PAD; j++) {
            for (int c = 0; c < CH; c++) {
                dout[i][j][c] = 0;
            }
        }
    }

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < CH; c++) {
                dout[i + PAD][j + PAD][c] = din[i][j][c];
            }
        }
    }
}
