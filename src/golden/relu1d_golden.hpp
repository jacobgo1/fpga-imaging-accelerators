#pragma once

// Elementwise ReLU over a [position][channel] spectrum (see conv1d_golden.hpp).
template<typename data_t, int L, int CH>
void relu1d_golden(const data_t din[L][CH], data_t dout[L][CH]) {
    for (int j = 0; j < L; j++) {
        for (int c = 0; c < CH; c++) {
            dout[j][c] = din[j][c] > 0 ? din[j][c] : 0;
        }
    }
}
