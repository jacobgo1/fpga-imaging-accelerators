#pragma once

// MaxPool1d(POOL) along the spectrum. As in PyTorch, a last window that would
// stick out is dropped: LENGTH / POOL positions out.
template<typename data_t, int LENGTH, int CHANNELS, int POOL>
void maxpool1d_golden(const data_t din[LENGTH][CHANNELS], data_t dout[LENGTH / POOL][CHANNELS]) {
    for (int pos = 0; pos < LENGTH / POOL; pos++)
        for (int ch = 0; ch < CHANNELS; ch++) {
            data_t largest = din[pos * POOL][ch];
            for (int p = 1; p < POOL; p++)
                if (din[pos * POOL + p][ch] > largest)
                    largest = din[pos * POOL + p][ch];
            dout[pos][ch] = largest;
        }
}
