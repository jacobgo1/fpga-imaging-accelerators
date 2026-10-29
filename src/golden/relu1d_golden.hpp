#pragma once

template<typename data_t, int LENGTH, int CHANNELS>
void relu1d_golden(const data_t din[LENGTH][CHANNELS], data_t dout[LENGTH][CHANNELS]) {
    for (int pos = 0; pos < LENGTH; pos++)
        for (int ch = 0; ch < CHANNELS; ch++)
            dout[pos][ch] = din[pos][ch] > 0 ? din[pos][ch] : 0;
}
