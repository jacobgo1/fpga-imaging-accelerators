#pragma once

// 1-level 1D discrete wavelet transform along the spectral bands of each pixel,
// concatenated onto the original bands, as in WaveletCNN (pytorch_wavelets
// DWT1DForward(J=1, wave='db4', mode='zero') followed by torch.cat([x, low, high])).
//
// With IN_CH = 110 bands and L = 8 filter taps (Daubechies-4):
//   SUB_CH = (IN_CH + L - 1) / 2 = 58 coefficients per subband,
//   PAD    = (2 * (SUB_CH - 1) - IN_CH + L) / 2 = 6 zero bands on each side,
//   CAT_CH = IN_CH + 2 * SUB_CH = 226 channels out per pixel:
//     channels 0 .. IN_CH - 1:               the input bands x,
//     channels IN_CH .. IN_CH + SUB_CH - 1:  lowpass  (x_pad cross-correlated with h0, stride 2),
//     channels IN_CH + SUB_CH .. CAT_CH - 1: highpass (x_pad cross-correlated with h1, stride 2).
template<typename data_t, int H, int W, int IN_CH, int L>
void dwt1d_concat_golden(
    const data_t din[H][W][IN_CH],
    const data_t h0[1][1][L],
    const data_t h1[1][1][L],
    data_t dout[H][W][IN_CH + 2 * ((IN_CH + L - 1) / 2)])
{
    constexpr int SUB_CH = (IN_CH + L - 1) / 2;
    constexpr int PAD = (2 * (SUB_CH - 1) - IN_CH + L) / 2;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < IN_CH; c++)
                dout[i][j][c] = din[i][j][c];
            for (int m = 0; m < SUB_CH; m++) {
                data_t lo = 0, hi = 0;
                for (int k = 0; k < L; k++) {
                    int idx = 2 * m + k - PAD;
                    if (idx < 0 || idx >= IN_CH) continue;
                    lo += din[i][j][idx] * h0[0][0][k];
                    hi += din[i][j][idx] * h1[0][0][k];
                }
                dout[i][j][IN_CH + m] = lo;
                dout[i][j][IN_CH + SUB_CH + m] = hi;
            }
        }
    }
}

template<typename act_t, typename w_t, int H, int W, int IN_CH, int L>
void dwt1d_concat_quant_golden(
    const act_t din[H][W][IN_CH],
    const w_t h0[1][1][L], act_t h0_scale,
    const w_t h1[1][1][L], act_t h1_scale,
    act_t dout[H][W][IN_CH + 2 * ((IN_CH + L - 1) / 2)])
{
    constexpr int SUB_CH = (IN_CH + L - 1) / 2;
    constexpr int PAD = (2 * (SUB_CH - 1) - IN_CH + L) / 2;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            for (int c = 0; c < IN_CH; c++)
                dout[i][j][c] = din[i][j][c];
            for (int m = 0; m < SUB_CH; m++) {
                act_t lo = 0, hi = 0;
                for (int k = 0; k < L; k++) {
                    int idx = 2 * m + k - PAD;
                    if (idx < 0 || idx >= IN_CH) continue;
                    lo += din[i][j][idx] * (act_t)h0[0][0][k];
                    hi += din[i][j][idx] * (act_t)h1[0][0][k];
                }
                dout[i][j][IN_CH + m] = lo * h0_scale;
                dout[i][j][IN_CH + SUB_CH + m] = hi * h1_scale;
            }
        }
    }
}
