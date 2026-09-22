#pragma once

// Elementwise addition of two equally-shaped tensors.
template<typename data_t, int H, int W, int CH>
void add_golden(data_t a[H][W][CH], data_t b[H][W][CH], data_t dout[H][W][CH]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < CH; c++) {
                dout[i][j][c] = a[i][j][c] + b[i][j][c];
            }
        }
    }
}
