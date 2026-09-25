#pragma once

// Fully connected layer (PyTorch nn.Linear), applied pointwise to every
// row of a [H][IN_FEATURES] tensor (the W axis has already been flattened
// away by flatten_golden by this point) -- same per-row independence as
// conv1x1_golden. Weight layout matches PyTorch's native Linear weight:
// [out_features][in_features].
template<typename data_t, int H, int IN_FEATURES, int OUT_FEATURES>
void dense_golden(data_t din[H][IN_FEATURES],
                   data_t weight[OUT_FEATURES][IN_FEATURES],
                   data_t bias[OUT_FEATURES],
                   data_t dout[H][OUT_FEATURES]) {
    for (int i = 0; i < H; i++) {
        for (int o = 0; o < OUT_FEATURES; o++) {
            data_t sum = bias[o];
            for (int f = 0; f < IN_FEATURES; f++) {
                sum += din[i][f] * weight[o][f];
            }
            dout[i][o] = sum;
        }
    }
}
