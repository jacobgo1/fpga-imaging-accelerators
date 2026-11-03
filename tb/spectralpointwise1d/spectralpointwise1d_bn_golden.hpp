#pragma once

#include "../golden/batchnorm.hpp"

// 1D-SpectralPointwise as trained, in float, with its BatchNorm: the testbench's
// reference for the golden kernel spectralpointwise1d. A 1x1 conv on one pixel is a
// dense layer.
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
    using golden::activation;
    data_t relu1[CONV1_CHANNELS], relu2[CONV2_CHANNELS];
    golden::dense<BANDS, CONV1_CHANNELS>(din, conv1_weight, conv1_bias, relu1);
    golden::batchnorm<CONV1_CHANNELS, activation::relu>(relu1, 1, bn_weight, bn_bias, bn_mean, bn_var, bn_eps);
    golden::dense<CONV1_CHANNELS, CONV2_CHANNELS, activation::relu>(relu1, conv2_weight, conv2_bias, relu2);
    golden::dense<CONV2_CHANNELS, CLASSES>(relu2, conv3_weight, conv3_bias, dout);
}
