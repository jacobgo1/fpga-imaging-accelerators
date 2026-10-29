#pragma once

// WaveletCNN's spectral transform: a one-level db4 wavelet transform along each
// pixel's bands (pytorch_wavelets DWT1DForward(J=1, wave='db4', mode='zero')),
// appended to the bands. For 110 bands and 8 taps: the 110 bands, 58 lowpass,
// 58 highpass, 226 channels out. The filters may be quantized: each sum is
// multiplied by its filter's scale (1 for float filters).
constexpr int dwt1d_subband_length(int bands, int taps) { return (bands + taps - 1) / 2; }
constexpr int dwt1d_output_channels(int bands, int taps) { return bands + 2 * dwt1d_subband_length(bands, taps); }

template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int BANDS, int TAPS>
void dwt1d_concat_golden(const act_t din[HEIGHT][WIDTH][BANDS],
                         const weight_t lowpass[1][1][TAPS], act_t lowpass_scale,
                         const weight_t highpass[1][1][TAPS], act_t highpass_scale,
                         act_t dout[HEIGHT][WIDTH][dwt1d_output_channels(BANDS, TAPS)]) {
    constexpr int SUBBAND = dwt1d_subband_length(BANDS, TAPS);
    constexpr int PAD = (2 * (SUBBAND - 1) - BANDS + TAPS) / 2;   // zero bands on each side
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++) {
            for (int band = 0; band < BANDS; band++)
                dout[row][col][band] = din[row][col][band];
            for (int m = 0; m < SUBBAND; m++) {
                act_t low = 0, high = 0;
                for (int tap = 0; tap < TAPS; tap++) {
                    const int band = 2 * m + tap - PAD;
                    if (band < 0 || band >= BANDS) continue;
                    low += din[row][col][band] * (act_t)lowpass[0][0][tap];
                    high += din[row][col][band] * (act_t)highpass[0][0][tap];
                }
                dout[row][col][BANDS + m] = low * lowpass_scale;
                dout[row][col][BANDS + SUBBAND + m] = high * highpass_scale;
            }
        }
}
