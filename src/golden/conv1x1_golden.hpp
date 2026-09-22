#pragma once

// 1x1 convolution: no spatial window, just a per-pixel channel mix.
// PyTorch's Conv2d(kernel_size=1) weight has shape [out][in][1][1]; squeeze
// the trailing singleton dims (pure reshape, no data movement) to get the
// [out][in] layout expected here.
template<typename data_t, int H, int W, int IN_CH, int OUT_CH>
void conv1x1_golden(data_t din[H][W][IN_CH],
                     data_t weight[OUT_CH][IN_CH],
                     data_t bias[OUT_CH],
                     data_t dout[H][W][OUT_CH]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int oc = 0; oc < OUT_CH; oc++) {
                data_t sum = bias[oc];
                for (int ic = 0; ic < IN_CH; ic++) {
                    sum += din[i][j][ic] * weight[oc][ic];
                }
                dout[i][j][oc] = sum;
            }
        }
    }
}
