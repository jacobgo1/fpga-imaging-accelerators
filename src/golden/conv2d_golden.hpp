#pragma once

// Valid convolution, stride 1, no padding: output is smaller than the input.
// Weight layout matches PyTorch's native Conv2d weight: [out][in][kh][kw].
template<typename data_t, int H, int W, int IN_CH, int OUT_CH, int K>
void conv2d_golden(data_t din[H][W][IN_CH],
                    data_t weight[OUT_CH][IN_CH][K][K],
                    data_t bias[OUT_CH],
                    data_t dout[H - K + 1][W - K + 1][OUT_CH]) {
    for (int i = 0; i < H - K + 1; i++) {
        for (int j = 0; j < W - K + 1; j++) {
            for (int oc = 0; oc < OUT_CH; oc++) {
                data_t sum = bias[oc];
                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {
                            sum += din[i + ki][j + kj][ic] * weight[oc][ic][ki][kj];
                        }
                    }
                }
                dout[i][j][oc] = sum;
            }
        }
    }
}
