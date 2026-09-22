#pragma once

template<typename data_t, int H, int W, int IM_CH, int POOL>
void maxpool_golden(data_t din[H][W][IM_CH], data_t dout[H/2][W/2][IM_CH]) {
    for (int i = 0; i < H; i += 2) {
        for (int j = 0; j < W; j += 2) {
            for (int k = 0; k < IM_CH; k++) {
                data_t max_val = din[i][j][k];
                for (int pool_i = 0; pool_i < POOL; pool_i++) {
                    for (int pool_j = 0; pool_j < POOL; pool_j++) {
                        if (din[i + pool_i][j + pool_j][k] > max_val) {
                            max_val = din[i + pool_i][j + pool_j][k];
                        }
                    }
                }
                dout[i/2][j/2][k] = max_val;
            }
        }
    }
}
