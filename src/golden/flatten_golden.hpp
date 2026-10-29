#pragma once

// torch.flatten(x, 1) on PyTorch's (channel, position) order: our spectra are
// [position][channel], so this transposes, and the dense weights load as trained.
template<typename data_t, int LENGTH, int CHANNELS>
void flatten_golden(const data_t din[LENGTH][CHANNELS], data_t dout[CHANNELS * LENGTH]) {
    for (int ch = 0; ch < CHANNELS; ch++)
        for (int pos = 0; pos < LENGTH; pos++)
            dout[ch * LENGTH + pos] = din[pos][ch];
}
