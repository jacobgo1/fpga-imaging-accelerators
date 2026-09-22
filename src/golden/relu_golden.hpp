#pragma once

template<typename data_t, int H, int W, int CH>
void relu_golden(data_t din[H][W][CH], data_t dout[H][W][CH]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < CH; c++) {
                dout[i][j][c] = din[i][j][c] > 0 ? din[i][j][c] : 0;
            }
        }
    }
}
