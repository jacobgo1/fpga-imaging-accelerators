#pragma once

#include "dense_golden.hpp"
#include "batchnorm1d_golden.hpp"
#include "relu1d_golden.hpp"

// 1D-SpectralPointwise as trained, in float, with its BatchNorm: the testbench's
// reference for spectralpointwise1d_golden.hpp. A 1x1 conv on one pixel is a
// dense layer; the pixel is a spectrum of one position for the BatchNorm and ReLUs.
template<typename data_t, int BANDS, int CONV1_CHANNELS, int CONV2_CHANNELS, int CLASSES>
void spectralpointwise1d_bn_golden(
    const data_t din[BANDS],
    const data_t conv1_weight[CONV1_CHANNELS][BANDS], const data_t conv1_bias[CONV1_CHANNELS],
    const data_t bn_weight[CONV1_CHANNELS],           const data_t bn_bias[CONV1_CHANNELS],
    const data_t bn_mean[CONV1_CHANNELS],             const data_t bn_var[CONV1_CHANNELS], data_t bn_eps,
    const data_t conv2_weight[CONV2_CHANNELS][CONV1_CHANNELS], const data_t conv2_bias[CONV2_CHANNELS],
    const data_t conv3_weight[CLASSES][CONV2_CHANNELS],        const data_t conv3_bias[CLASSES],
    data_t dout[CLASSES])
{
    data_t conv1[1][CONV1_CHANNELS], normalized1[1][CONV1_CHANNELS], relu1[1][CONV1_CHANNELS];
    data_t conv2[1][CONV2_CHANNELS], relu2[1][CONV2_CHANNELS];
    dense_golden<data_t, BANDS, CONV1_CHANNELS>(din, conv1_weight, conv1_bias, conv1[0]);
    batchnorm1d_golden<data_t, 1, CONV1_CHANNELS>(conv1, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, normalized1);
    relu1d_golden<data_t, 1, CONV1_CHANNELS>(normalized1, relu1);
    dense_golden<data_t, CONV1_CHANNELS, CONV2_CHANNELS>(relu1[0], conv2_weight, conv2_bias, conv2[0]);
    relu1d_golden<data_t, 1, CONV2_CHANNELS>(conv2, relu2);
    dense_golden<data_t, CONV2_CHANNELS, CLASSES>(relu2[0], conv3_weight, conv3_bias, dout);
}
