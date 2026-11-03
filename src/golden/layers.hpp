#pragma once

// The golden kernels' layers: plain loops, what is computed. Feature maps are
// [row][col][channel], spectra [position][channel], weights in PyTorch's layouts.
//
// The same layer serves float and quantized weights. Each output is
//     sum(x * w) * weight_scale + bias * bias_scale, then the activation,
// so float weights leave the scales at 1, and quantized ones are the int8 q
// (or floats holding them) with their 2^-frac.

namespace golden {

// The quantized weights are q * 2^-frac: this is that 2^-frac.
constexpr float frac_to_scale(int frac) {
    return 1.0f / (float)(1L << frac);
}

enum class activation { none, relu, leaky_relu };

constexpr float LEAKY_SLOPE = 0.01f;   // WaveletCNN's LeakyReLU

template<activation A, typename T>
T activate(T x){ 
    if (A == activation::relu) return x < 0 ? (T)0 : x;
    if (A == activation::leaky_relu) return x > 0 ? x : (T)LEAKY_SLOPE * x;
    return x;
}

// Conv2d with zero padding (K - 1) / 2, so the output is the input's size.
template<int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL,
         activation A = activation::none, typename T, typename W>
void conv2d(const T din[HEIGHT][WIDTH][IN_CHANNELS],
            const W weight[OUT_CHANNELS][IN_CHANNELS][KERNEL][KERNEL], const W bias[OUT_CHANNELS],
            T dout[HEIGHT][WIDTH][OUT_CHANNELS],
            T weight_scale = 1, T bias_scale = 1) {
    constexpr int PAD = (KERNEL - 1) / 2;
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            for (int out = 0; out < OUT_CHANNELS; out++) {
                T sum = 0;
                for (int in = 0; in < IN_CHANNELS; in++)
                    for (int ky = 0; ky < KERNEL; ky++)
                        for (int kx = 0; kx < KERNEL; kx++) {
                            const int y = row + ky - PAD, x = col + kx - PAD;
                            if (y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH)
                                sum += din[y][x][in] * (T)weight[out][in][ky][kx];
                        }
                dout[row][col][out] = activate<A>(sum * weight_scale + (T)bias[out] * bias_scale);
            }
}

// Conv1d without padding: LENGTH positions in, LENGTH - KERNEL + 1 out.
template<int LENGTH, int IN_CHANNELS, int OUT_CHANNELS, int KERNEL,
         activation A = activation::none, typename T, typename W>
void conv1d(const T din[LENGTH][IN_CHANNELS],
            const W weight[OUT_CHANNELS][IN_CHANNELS][KERNEL], const W bias[OUT_CHANNELS],
            T dout[LENGTH - KERNEL + 1][OUT_CHANNELS],
            T weight_scale = 1, T bias_scale = 1) {
    for (int pos = 0; pos < LENGTH - KERNEL + 1; pos++)
        for (int out = 0; out < OUT_CHANNELS; out++) {
            T sum = 0;
            for (int in = 0; in < IN_CHANNELS; in++)
                for (int k = 0; k < KERNEL; k++)
                    sum += din[pos + k][in] * (T)weight[out][in][k];
            dout[pos][out] = activate<A>(sum * weight_scale + (T)bias[out] * bias_scale);
        }
}

// nn.Linear.
template<int INPUTS, int OUTPUTS, activation A = activation::none, typename T, typename W>
void dense(const T din[INPUTS], const W weight[OUTPUTS][INPUTS], const W bias[OUTPUTS],
           T dout[OUTPUTS],
           T weight_scale = 1, T bias_scale = 1) {
    for (int out = 0; out < OUTPUTS; out++) {
        T sum = 0;
        for (int in = 0; in < INPUTS; in++)
            sum += din[in] * (T)weight[out][in];
        dout[out] = activate<A>(sum * weight_scale + (T)bias[out] * bias_scale);
    }
}

// 2 x 2 max pooling, stride 2.
template<int HEIGHT, int WIDTH, int CHANNELS, typename T>
void maxpool2d(const T din[HEIGHT][WIDTH][CHANNELS], T dout[HEIGHT / 2][WIDTH / 2][CHANNELS]) {
    for (int row = 0; row < HEIGHT / 2; row++)
        for (int col = 0; col < WIDTH / 2; col++)
            for (int ch = 0; ch < CHANNELS; ch++) {
                T largest = din[2 * row][2 * col][ch];
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                        if (din[2 * row + dy][2 * col + dx][ch] > largest)
                            largest = din[2 * row + dy][2 * col + dx][ch];
                dout[row][col][ch] = largest;
            }
}

// MaxPool1d(POOL). As in PyTorch, a last window that would stick out is dropped.
template<int LENGTH, int CHANNELS, int POOL, typename T>
void maxpool1d(const T din[LENGTH][CHANNELS], T dout[LENGTH / POOL][CHANNELS]) {
    for (int pos = 0; pos < LENGTH / POOL; pos++)
        for (int ch = 0; ch < CHANNELS; ch++) {
            T largest = din[pos * POOL][ch];
            for (int p = 1; p < POOL; p++)
                if (din[pos * POOL + p][ch] > largest)
                    largest = din[pos * POOL + p][ch];
            dout[pos][ch] = largest;
        }
}

// Nearest-neighbour upsampling by 2: each pixel becomes a 2 x 2 block.
template<int HEIGHT, int WIDTH, int CHANNELS, typename T>
void upsample2d(const T din[HEIGHT][WIDTH][CHANNELS], T dout[2 * HEIGHT][2 * WIDTH][CHANNELS]) {
    for (int row = 0; row < 2 * HEIGHT; row++)
        for (int col = 0; col < 2 * WIDTH; col++)
            for (int ch = 0; ch < CHANNELS; ch++)
                dout[row][col][ch] = din[row / 2][col / 2][ch];
}

// torch.flatten(x, 1) on PyTorch's (channel, position) order: our spectra are
// [position][channel], so this transposes, and the dense weights load as trained.
template<int LENGTH, int CHANNELS, typename T>
void flatten(const T din[LENGTH][CHANNELS], T dout[CHANNELS * LENGTH]) {
    for (int ch = 0; ch < CHANNELS; ch++)
        for (int pos = 0; pos < LENGTH; pos++)
            dout[ch * LENGTH + pos] = din[pos][ch];
}

// torch.cat([first, second], dim=1): each pixel's channels of first, then of second.
// The order matters: SpecPW-UNet's accuracy drops from 95 % to 30-40 % the other way round.
template<int HEIGHT, int WIDTH, int FIRST_CHANNELS, int SECOND_CHANNELS, typename T>
void concat(const T first[HEIGHT][WIDTH][FIRST_CHANNELS], const T second[HEIGHT][WIDTH][SECOND_CHANNELS],
            T dout[HEIGHT][WIDTH][FIRST_CHANNELS + SECOND_CHANNELS]) {
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++) {
            for (int ch = 0; ch < FIRST_CHANNELS; ch++)
                dout[row][col][ch] = first[row][col][ch];
            for (int ch = 0; ch < SECOND_CHANNELS; ch++)
                dout[row][col][FIRST_CHANNELS + ch] = second[row][col][ch];
        }
}

// WaveletCNN's spectral transform: a one-level db4 wavelet transform along each
// pixel's bands (pytorch_wavelets DWT1DForward(J=1, wave='db4', mode='zero')),
// appended to the bands. For 110 bands and 8 taps: the 110 bands, 58 lowpass,
// 58 highpass, 226 channels out.
constexpr int dwt1d_subband_length(int bands, int taps) { return (bands + taps - 1) / 2; }
constexpr int dwt1d_output_channels(int bands, int taps) { return bands + 2 * dwt1d_subband_length(bands, taps); }

template<int HEIGHT, int WIDTH, int BANDS, int TAPS, typename T, typename W>
void dwt1d_concat(const T din[HEIGHT][WIDTH][BANDS],
                  const W lowpass[1][1][TAPS], const W highpass[1][1][TAPS],
                  T dout[HEIGHT][WIDTH][dwt1d_output_channels(BANDS, TAPS)],
                  T lowpass_scale = 1, T highpass_scale = 1) {
    constexpr int SUBBAND = dwt1d_subband_length(BANDS, TAPS);
    constexpr int PAD = (2 * (SUBBAND - 1) - BANDS + TAPS) / 2;   // zero bands on each side
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++) {
            for (int band = 0; band < BANDS; band++)
                dout[row][col][band] = din[row][col][band];
            for (int m = 0; m < SUBBAND; m++) {
                T low = 0, high = 0;
                for (int tap = 0; tap < TAPS; tap++) {
                    const int band = 2 * m + tap - PAD;
                    if (band < 0 || band >= BANDS) continue;
                    low += din[row][col][band] * (T)lowpass[0][0][tap];
                    high += din[row][col][band] * (T)highpass[0][0][tap];
                }
                dout[row][col][BANDS + m] = low * lowpass_scale;
                dout[row][col][BANDS + SUBBAND + m] = high * highpass_scale;
            }
        }
}

}  // namespace golden
