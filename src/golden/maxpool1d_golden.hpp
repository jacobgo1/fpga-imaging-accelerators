#pragma once

// Non-overlapping max pooling (PyTorch's MaxPool1d with stride == kernel
// size, e.g. nn.MaxPool1d(2)) along W only, H untouched (see
// conv1d_golden.hpp): window POOL, stride POOL -- one fewer window loop
// than maxpool_golden's 2D pooling (a single p loop instead of pool_i/pool_j).
template<typename data_t, int H, int W, int CH, int POOL>
void maxpool1d_golden(data_t din[H][W][CH], data_t dout[H][W / POOL][CH]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j += POOL) {
            for (int c = 0; c < CH; c++) {
                data_t max_val = din[i][j][c];
                for (int p = 1; p < POOL; p++) {
                    if (din[i][j + p][c] > max_val) {
                        max_val = din[i][j + p][c];
                    }
                }
                dout[i][j / POOL][c] = max_val;
            }
        }
    }
}
