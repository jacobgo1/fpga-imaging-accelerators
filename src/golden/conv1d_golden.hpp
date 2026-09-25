#pragma once

// Pointwise 1D convolution over a WxH picture: same [H][W][IN_CH] shape as
// conv2d_golden, but the kernel is 1D (size K, not KxK) and slides only
// along W. H just replicates independent rows/pixels -- no mixing across
// H, same as conv2d_golden has no mixing across OUT_CH -- so there is one
// fewer loop here than in conv2d_golden: one kernel-window loop (k) instead
// of two (ki, kj).
//
// Weight layout matches PyTorch's native Conv1d weight: [out][in][k].
template<typename data_t, int H, int W, int IN_CH, int OUT_CH, int K>
void conv1d_golden(data_t din[H][W][IN_CH],
                    data_t weight[OUT_CH][IN_CH][K],
                    data_t bias[OUT_CH],
                    data_t dout[H][W - K + 1][OUT_CH]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W - K + 1; j++) {
            for (int oc = 0; oc < OUT_CH; oc++) {
                data_t sum = bias[oc];
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int k = 0; k < K; k++) {
                        sum += din[i][j + k][ic] * weight[oc][ic][k];
                    }
                }
                dout[i][j][oc] = sum;
            }
        }
    }
}
