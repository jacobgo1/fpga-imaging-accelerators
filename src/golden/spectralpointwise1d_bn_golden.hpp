#pragma once

#include "dense_golden.hpp"
#include "batchnorm1d_golden.hpp"
#include "relu1d_golden.hpp"

// 1D-SpectralPointwise as trained, in float, from the plain modules: the
// PyTorch model SpectralPointwise1D, for one pixel. It is the testbench's
// reference for the quantized spectralpointwise1d_golden.hpp, as
// justoliunet_bn_golden.hpp is for justoliunet:
//
//   x = relu(bnsp1(convsp1(x)))          bnsp1: BatchNorm2d(CH1) on a 1 x 1 image
//   x = relu(convsp2(x))
//   x = convsp3(x)
//
// A 1x1 convolution on one pixel is a dense layer, so the convs are
// dense_golden with the Conv2d weights [out][in][1][1] read as [out][in]; a
// pixel is a spectrum of one position for the BatchNorm and the ReLUs. With an
// identity BatchNorm (weight 1, bias 0, mean 0, var 1, eps 0) it is the network
// with the BatchNorm folded in, as the quantized kernel has it.
template<typename data_t, int BANDS, int CH1, int CH2, int CLASSES>
void spectralpointwise1d_bn_golden(
    const data_t din[BANDS],
    const data_t w1[CH1][BANDS],   const data_t b1[CH1],
    const data_t bn_weight[CH1],   const data_t bn_bias[CH1],
    const data_t bn_mean[CH1],     const data_t bn_var[CH1], data_t bn_eps,
    const data_t w2[CH2][CH1],     const data_t b2[CH2],
    const data_t w3[CLASSES][CH2], const data_t b3[CLASSES],
    data_t dout[CLASSES])
{
    data_t c1[1][CH1], bn1[1][CH1], r1[1][CH1], c2[1][CH2], r2[1][CH2];
    dense_golden<data_t, BANDS, CH1>(din, w1, b1, c1[0]);
    batchnorm1d_golden<data_t, 1, CH1>(c1, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, bn1);
    relu1d_golden<data_t, 1, CH1>(bn1, r1);
    dense_golden<data_t, CH1, CH2>(r1[0], w2, b2, c2[0]);
    relu1d_golden<data_t, 1, CH2>(c2, r2);
    dense_golden<data_t, CH2, CLASSES>(r2[0], w3, b3, dout);
}
