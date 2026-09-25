#pragma once

#include "conv1d_golden.hpp"
#include "relu1d_golden.hpp"
#include "maxpool1d_golden.hpp"
#include "flatten_golden.hpp"
#include "dense_golden.hpp"

// 1D-Justo-LiuNet, run pointwise over a WxH picture (see
// conv1d_golden.hpp): every row is one independent length-W spectrum, and
// the same four (conv1d -> relu -> maxpool1d(2)) stages -- widening the
// channel count k -> 2k -> 3k -> 4k while W shrinks -- are applied
// independently and identically to every row, followed by flatten and a
// dense head, producing one class-score vector per row. No row's
// computation ever depends on another row's data; H just says how many
// spectra this call classifies at once. This mirrors the following
// PyTorch model, applied once per row:
//
//   self.conv1 = nn.Conv1d(1, k, kernel_size=K);    self.pool1 = nn.MaxPool1d(2)
//   self.conv2 = nn.Conv1d(k, 2k, kernel_size=K);   self.pool2 = nn.MaxPool1d(2)
//   self.conv3 = nn.Conv1d(2k, 3k, kernel_size=K);  self.pool3 = nn.MaxPool1d(2)
//   self.conv4 = nn.Conv1d(3k, 4k, kernel_size=K);  self.pool4 = nn.MaxPool1d(2)
//   x = flatten(x, 1); x = self.fc(x)
//
// Every weight/bias pair matches PyTorch's native Conv1d/Linear parameters
// ([out][in][k] + [out], [out][in] + [out]) so a trained state_dict loads
// directly -- weights are shared across every row, same as a real
// convolution's weights are shared across an image. Only the activations
// differ in axis order from PyTorch's channel-first spectra; flatten_golden
// already accounts for that for the vector handed to the dense layer.
//
// W must survive four rounds of (conv, kernel K) then (pool, /2) without
// hitting zero or a non-multiple-of-2 length: a mismatch fails to compile
// (a negative or zero array bound), not silently misbehave.
constexpr int justoliunet_conv_len(int len_in, int K) { return len_in - K + 1; }
constexpr int justoliunet_pool_len(int len_in) { return len_in / 2; }
constexpr int justoliunet_flatten_dim(int W, int K, int BASE_CH) {
    return 4 * BASE_CH *
        justoliunet_pool_len(justoliunet_conv_len(
        justoliunet_pool_len(justoliunet_conv_len(
        justoliunet_pool_len(justoliunet_conv_len(
        justoliunet_pool_len(justoliunet_conv_len(W, K)), K)), K)), K));
}

template<typename data_t, int H, int W, int K, int BASE_CH, int NUM_CLASSES>
void justoliunet_golden(
    data_t din[H][W][1],
    data_t w1[BASE_CH][1][K],           data_t b1[BASE_CH],
    data_t w2[2*BASE_CH][BASE_CH][K],   data_t b2[2*BASE_CH],
    data_t w3[3*BASE_CH][2*BASE_CH][K], data_t b3[3*BASE_CH],
    data_t w4[4*BASE_CH][3*BASE_CH][K], data_t b4[4*BASE_CH],
    data_t w5[NUM_CLASSES][justoliunet_flatten_dim(W, K, BASE_CH)], data_t b5[NUM_CLASSES],
    data_t dout[H][NUM_CLASSES])
{
    constexpr int W1C = W   - K + 1, W1P = W1C / 2;
    constexpr int W2C = W1P - K + 1, W2P = W2C / 2;
    constexpr int W3C = W2P - K + 1, W3P = W3C / 2;
    constexpr int W4C = W3P - K + 1, W4P = W4C / 2;
    constexpr int FLAT = 4 * BASE_CH * W4P;

    data_t conv1_out[H][W1C][BASE_CH],   relu1_out[H][W1C][BASE_CH],   pool1_out[H][W1P][BASE_CH];
    data_t conv2_out[H][W2C][2*BASE_CH], relu2_out[H][W2C][2*BASE_CH], pool2_out[H][W2P][2*BASE_CH];
    data_t conv3_out[H][W3C][3*BASE_CH], relu3_out[H][W3C][3*BASE_CH], pool3_out[H][W3P][3*BASE_CH];
    data_t conv4_out[H][W4C][4*BASE_CH], relu4_out[H][W4C][4*BASE_CH], pool4_out[H][W4P][4*BASE_CH];
    data_t flat_out[H][FLAT];

    conv1d_golden<data_t, H, W, 1, BASE_CH, K>(din, w1, b1, conv1_out);
    relu1d_golden<data_t, H, W1C, BASE_CH>(conv1_out, relu1_out);
    maxpool1d_golden<data_t, H, W1C, BASE_CH, 2>(relu1_out, pool1_out);

    conv1d_golden<data_t, H, W1P, BASE_CH, 2*BASE_CH, K>(pool1_out, w2, b2, conv2_out);
    relu1d_golden<data_t, H, W2C, 2*BASE_CH>(conv2_out, relu2_out);
    maxpool1d_golden<data_t, H, W2C, 2*BASE_CH, 2>(relu2_out, pool2_out);

    conv1d_golden<data_t, H, W2P, 2*BASE_CH, 3*BASE_CH, K>(pool2_out, w3, b3, conv3_out);
    relu1d_golden<data_t, H, W3C, 3*BASE_CH>(conv3_out, relu3_out);
    maxpool1d_golden<data_t, H, W3C, 3*BASE_CH, 2>(relu3_out, pool3_out);

    conv1d_golden<data_t, H, W3P, 3*BASE_CH, 4*BASE_CH, K>(pool3_out, w4, b4, conv4_out);
    relu1d_golden<data_t, H, W4C, 4*BASE_CH>(conv4_out, relu4_out);
    maxpool1d_golden<data_t, H, W4C, 4*BASE_CH, 2>(relu4_out, pool4_out);

    flatten_golden<data_t, H, W4P, 4*BASE_CH>(pool4_out, flat_out);
    dense_golden<data_t, H, FLAT, NUM_CLASSES>(flat_out, w5, b5, dout);
}
