#pragma once

#include <cmath>

// PyTorch's BatchNorm2d in inference mode: a per-channel affine transform
// using running statistics, nothing computed from din itself (batch
// mean/var only ever get computed during training). weight is PyTorch's
// gamma, bias is its beta; weight, bias, running_mean and running_var are
// all length-CH vectors, the same layout BatchNorm2d's state_dict stores
// them in, so a trained model's affine params and running stats load
// directly with no reshuffling. eps is passed in, not hardcoded, so this
// matches whatever eps the source model was trained with (PyTorch's
// default is 1e-5).
template<typename data_t, int H, int W, int CH>
void batchnorm_golden(data_t din[H][W][CH],
                       data_t weight[CH], data_t bias[CH],
                       data_t running_mean[CH], data_t running_var[CH],
                       data_t eps,
                       data_t dout[H][W][CH]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < CH; c++) {
                data_t normalized = (din[i][j][c] - running_mean[c]) / std::sqrt(running_var[c] + eps);
                dout[i][j][c] = weight[c] * normalized + bias[c];
            }
        }
    }
}
