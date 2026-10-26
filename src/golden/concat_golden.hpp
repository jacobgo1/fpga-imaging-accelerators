#pragma once

// A U-Net's skip connection: PyTorch's torch.cat([a, b], dim=1) on HWC maps.
// Each pixel's output channels are a's CH_A channels, then b's CH_B. The order
// matters: the next conv's weights expect it (SpecPW-UNet concatenates the
// upsampled map first, then the skip; the other order drops its accuracy on the
// labelled aegean pixels from 94-96 % to 27-40 %).
template<typename data_t, int H, int W, int CH_A, int CH_B>
void concat_golden(const data_t a[H][W][CH_A],
                   const data_t b[H][W][CH_B],
                   data_t dout[H][W][CH_A + CH_B]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < CH_A; c++) {
                dout[i][j][c] = a[i][j][c];
            }
            for (int c = 0; c < CH_B; c++) {
                dout[i][j][CH_A + c] = b[i][j][c];
            }
        }
    }
}
