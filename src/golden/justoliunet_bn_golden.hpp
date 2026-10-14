#pragma once

#include "conv1d_golden.hpp"
#include "batchnorm1d_golden.hpp"
#include "relu1d_golden.hpp"
#include "maxpool1d_golden.hpp"
#include "flatten_golden.hpp"
#include "dense_golden.hpp"

// 1D-Justo-LiuNet as trained, in float, from the plain modules: the PyTorch
// model justoliunet_bn, for one pixel. It is the testbenches' reference for the
// quantized justoliunet_golden.hpp, as justounet2d_golden.hpp is for
// justounetsimple:
//
//   x = pool(relu(bn1(conv1(x))))          bn1: BatchNorm1d(BASE_CH)
//   x = pool(relu(conv2(x)))
//   x = pool(relu(conv3(x)))
//   x = pool(relu(conv4(x)))
//   x = fc(flatten(x, 1))
//
// With an identity BatchNorm (weight 1, bias 0, mean 0, var 1, eps 0) it is
// justoliunet, the model without BatchNorm. Every parameter has PyTorch's own
// layout (Conv1d [out][in][k], Linear [out][in], BatchNorm vectors of BASE_CH),
// so a state_dict loads directly. Activations are [position][channel].
template<typename data_t, int L, int BASE_CH, int CLASSES, int K>
void justoliunet_bn_golden(
    const data_t din[L][1],
    const data_t w1[BASE_CH][1][K],           const data_t b1[BASE_CH],
    const data_t bn_weight[BASE_CH],          const data_t bn_bias[BASE_CH],
    const data_t bn_mean[BASE_CH],            const data_t bn_var[BASE_CH], data_t bn_eps,
    const data_t w2[2*BASE_CH][BASE_CH][K],   const data_t b2[2*BASE_CH],
    const data_t w3[3*BASE_CH][2*BASE_CH][K], const data_t b3[3*BASE_CH],
    const data_t w4[4*BASE_CH][3*BASE_CH][K], const data_t b4[4*BASE_CH],
    const data_t wf[CLASSES][4*BASE_CH * (((((L - K + 1) / 2 - K + 1) / 2 - K + 1) / 2 - K + 1) / 2)],
    const data_t bf[CLASSES],
    data_t dout[CLASSES])
{
    constexpr int C1 = L  - K + 1, P1 = C1 / 2;
    constexpr int C2 = P1 - K + 1, P2 = C2 / 2;
    constexpr int C3 = P2 - K + 1, P3 = C3 / 2;
    constexpr int C4 = P3 - K + 1, P4 = C4 / 2;
    constexpr int FLAT = 4 * BASE_CH * P4;
    static_assert(P4 >= 1, "the spectrum is too short for four conv + pool blocks");

    data_t conv1[C1][BASE_CH], bn1[C1][BASE_CH], relu1[C1][BASE_CH], pool1[P1][BASE_CH];
    conv1d_golden<data_t, L, 1, BASE_CH, K>(din, w1, b1, conv1);
    batchnorm1d_golden<data_t, C1, BASE_CH>(conv1, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, bn1);
    relu1d_golden<data_t, C1, BASE_CH>(bn1, relu1);
    maxpool1d_golden<data_t, C1, BASE_CH, 2>(relu1, pool1);

    data_t conv2[C2][2*BASE_CH], relu2[C2][2*BASE_CH], pool2[P2][2*BASE_CH];
    conv1d_golden<data_t, P1, BASE_CH, 2*BASE_CH, K>(pool1, w2, b2, conv2);
    relu1d_golden<data_t, C2, 2*BASE_CH>(conv2, relu2);
    maxpool1d_golden<data_t, C2, 2*BASE_CH, 2>(relu2, pool2);

    data_t conv3[C3][3*BASE_CH], relu3[C3][3*BASE_CH], pool3[P3][3*BASE_CH];
    conv1d_golden<data_t, P2, 2*BASE_CH, 3*BASE_CH, K>(pool2, w3, b3, conv3);
    relu1d_golden<data_t, C3, 3*BASE_CH>(conv3, relu3);
    maxpool1d_golden<data_t, C3, 3*BASE_CH, 2>(relu3, pool3);

    data_t conv4[C4][4*BASE_CH], relu4[C4][4*BASE_CH], pool4[P4][4*BASE_CH];
    conv1d_golden<data_t, P3, 3*BASE_CH, 4*BASE_CH, K>(pool3, w4, b4, conv4);
    relu1d_golden<data_t, C4, 4*BASE_CH>(conv4, relu4);
    maxpool1d_golden<data_t, C4, 4*BASE_CH, 2>(relu4, pool4);

    data_t flat[FLAT];
    flatten_golden<data_t, P4, 4*BASE_CH>(pool4, flat);
    dense_golden<data_t, FLAT, CLASSES>(flat, wf, bf, dout);
}
