#pragma once

// PyTorch's torch.flatten(x, 1) on a channels-first (C, L) spectrum: element
// c * L + j is channel c at position j (channel-major). Our activations are
// [position][channel] (see conv1d_golden.hpp), so the transpose happens here,
// in the order of the two loops, and the dense layer's trained weights load
// unchanged.
template<typename data_t, int L, int CH>
void flatten_golden(const data_t din[L][CH], data_t dout[CH * L]) {
    for (int c = 0; c < CH; c++) {
        for (int j = 0; j < L; j++) {
            dout[c * L + j] = din[j][c];
        }
    }
}
