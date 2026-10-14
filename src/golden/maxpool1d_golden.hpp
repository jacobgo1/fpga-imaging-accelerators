#pragma once

// PyTorch's nn.MaxPool1d(POOL): windows of POOL, stride POOL, along the
// spectrum, per channel. Like PyTorch it drops a last window that would stick
// out: L in, L / POOL out (105 -> 52 in 1D-Justo-LiuNet's first block). The
// old version of this module walked to L and so read one position past the
// end whenever L was odd.
template<typename data_t, int L, int CH, int POOL>
void maxpool1d_golden(const data_t din[L][CH], data_t dout[L / POOL][CH]) {
    static_assert(L >= POOL, "the spectrum must hold at least one window");
    for (int j = 0; j < L / POOL; j++) {
        for (int c = 0; c < CH; c++) {
            data_t max_val = din[j * POOL][c];
            for (int p = 1; p < POOL; p++) {
                if (din[j * POOL + p][c] > max_val) {
                    max_val = din[j * POOL + p][c];
                }
            }
            dout[j][c] = max_val;
        }
    }
}
