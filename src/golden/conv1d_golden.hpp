#pragma once

// Valid (unpadded) 1D convolution along a spectrum: PyTorch's
// nn.Conv1d(IN_CH, OUT_CH, K) with stride 1 and padding 0, so L positions in,
// L - K + 1 out. The 1D networks convolve along the bands of one pixel
// (spectrally), never across pixels.
//
// Activations are [position][channel]: the spectrum's position first, the
// channels innermost, the same order as HWC everywhere else. Weights are
// PyTorch's native Conv1d layout, [out][in][k], so a state_dict loads as is.
template<typename data_t, int L, int IN_CH, int OUT_CH, int K>
void conv1d_golden(const data_t din[L][IN_CH],
                   const data_t weight[OUT_CH][IN_CH][K],
                   const data_t bias[OUT_CH],
                   data_t dout[L - K + 1][OUT_CH]) {
    static_assert(L >= K, "the spectrum must be at least as long as the kernel");
    for (int j = 0; j < L - K + 1; j++) {
        for (int oc = 0; oc < OUT_CH; oc++) {
            data_t sum = bias[oc];
            for (int ic = 0; ic < IN_CH; ic++) {
                for (int k = 0; k < K; k++) {
                    sum += din[j + k][ic] * weight[oc][ic][k];
                }
            }
            dout[j][oc] = sum;
        }
    }
}
