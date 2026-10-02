#pragma once

#include "conv2d_padded_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"
#include "maxpool_golden.hpp"
#include "nearest_neighbor_golden.hpp"

// 2D-JustoUNet: four same-padded 3x3-conv blocks, each pairing a Conv2d
// with BatchNorm2d (+ReLU on the first three, BatchNorm alone on the
// last), with down/upsampling folded into the middle two blocks instead
// of a separate symmetric encoder/decoder:
//
//   block A: conv(IN_CH->BASE_CH)      -> BN+ReLU -> maxpool(2x2)
//   block B: conv(BASE_CH->2*BASE_CH)  -> BN+ReLU -> maxpool(2x2) -> nearest upsample(2x)
//   block C: conv(2*BASE_CH->BASE_CH)  -> BN+ReLU ->                 nearest upsample(2x)
//   block D: conv(BASE_CH->OUT_CH)     -> BN
//
// Block A halves H,W once; block B halves them again internally and then
// immediately undoes it with a matching upsample, so B's output is back
// at A's resolution (H/2, W/2) but has looked at BASE_CH->2*BASE_CH
// features at the quarter-resolution bottleneck in between. Block C's
// upsample is what undoes block A's halving, bringing the map back to the
// original H, W before the final 1-conv head (block D) scores every
// pixel into OUT_CH channels.
//
// Every conv weight/bias pair matches PyTorch's native Conv2d parameters
// ([out][in][kh][kw] + [out]), and every BatchNorm2d weight/bias/
// running_mean/running_var quadruple matches its native per-channel
// layout ([out] each) -- a trained state_dict loads directly, no
// reshuffling. As with every other file here, activations are HWC
// (channels last); only PyTorch's own NCHW tensors need transposing
// (din/dout, not weights) to compare against this.
//
// This mirrors the following PyTorch model (eval mode -- running stats,
// no batch stats -- same as every batchnorm_golden call here):
//
//   self.conv1 = nn.Conv2d(IN_CH, BASE_CH, K, padding=(K-1)//2)
//   self.bn1   = nn.BatchNorm2d(BASE_CH)
//   self.conv2 = nn.Conv2d(BASE_CH, 2*BASE_CH, K, padding=(K-1)//2)
//   self.bn2   = nn.BatchNorm2d(2*BASE_CH)
//   self.conv3 = nn.Conv2d(2*BASE_CH, BASE_CH, K, padding=(K-1)//2)
//   self.bn3   = nn.BatchNorm2d(BASE_CH)
//   self.conv4 = nn.Conv2d(BASE_CH, OUT_CH, K, padding=(K-1)//2)
//   self.bn4   = nn.BatchNorm2d(OUT_CH)
//
//   x = relu(self.bn1(self.conv1(x))); x = maxpool2d(x, 2)             # block A
//   x = relu(self.bn2(self.conv2(x))); x = maxpool2d(x, 2); x = upsample(x, 2)  # block B
//   x = relu(self.bn3(self.conv3(x))); x = upsample(x, 2)              # block C
//   x = self.bn4(self.conv4(x))                                        # block D (head)
//
// wN/bN <- convN.weight/convN.bias; bnN_weight/bias/mean/var <-
// bnN.weight/bias/running_mean/running_var (see export_justounet2d_weights.py).
//
// H and W must be divisible by 4 (block A's halving, then block B's
// halving of what's left): sizes that aren't fail to compile, not
// silently misbehave, via the static_asserts below.
//
// Intermediate feature maps are "static": large ones don't belong on the
// call stack (see half_unet_golden.hpp).
template<typename data_t, int H, int W, int IN_CH, int BASE_CH, int OUT_CH, int K>
void justounet2d_golden(
    data_t din[H][W][IN_CH],
    data_t w1[BASE_CH][IN_CH][K][K],
    data_t b1[BASE_CH],
    data_t bn1_weight[BASE_CH], data_t bn1_bias[BASE_CH],
    data_t bn1_mean[BASE_CH],   data_t bn1_var[BASE_CH],
    data_t w2[2*BASE_CH][BASE_CH][K][K],
    data_t b2[2*BASE_CH],
    data_t bn2_weight[2*BASE_CH], data_t bn2_bias[2*BASE_CH],
    data_t bn2_mean[2*BASE_CH],   data_t bn2_var[2*BASE_CH],
    data_t w3[BASE_CH][2*BASE_CH][K][K],
    data_t b3[BASE_CH],
    data_t bn3_weight[BASE_CH], data_t bn3_bias[BASE_CH],
    data_t bn3_mean[BASE_CH],   data_t bn3_var[BASE_CH],
    data_t w4[OUT_CH][BASE_CH][K][K],
    data_t b4[OUT_CH],
    data_t bn4_weight[OUT_CH], data_t bn4_bias[OUT_CH],
    data_t bn4_mean[OUT_CH],   data_t bn4_var[OUT_CH],
    data_t eps,
    data_t dout[H][W][OUT_CH])
{
    static_assert(H % 4 == 0, "H must be divisible by 4 (block A's halving, then block B's)");
    static_assert(W % 4 == 0, "W must be divisible by 4 (block A's halving, then block B's)");

    constexpr int PAD = (K - 1) / 2;
    constexpr int H2 = H / 2, W2 = W / 2;  // after block A's maxpool
    constexpr int H4 = H2 / 2, W4 = W2 / 2;  // after block B's inner maxpool

    // ---- block A: conv -> BN+ReLU -> maxpool(2x2) ----
    static data_t convA[H][W][BASE_CH];
    static data_t bnA[H][W][BASE_CH];
    static data_t reluA[H][W][BASE_CH];
    static data_t poolA[H2][W2][BASE_CH];

    conv2d_padded_golden<data_t, H, W, IN_CH, BASE_CH, K, PAD>(din, w1, b1, convA);
    batchnorm_golden<data_t, H, W, BASE_CH>(convA, bn1_weight, bn1_bias, bn1_mean, bn1_var, eps, bnA);
    relu_golden<data_t, H, W, BASE_CH>(bnA, reluA);
    maxpool_golden<data_t, H, W, BASE_CH, 2>(reluA, poolA);

    // ---- block B: conv -> BN+ReLU -> maxpool(2x2) -> nearest upsample(2x) ----
    static data_t convB[H2][W2][2*BASE_CH];
    static data_t bnB[H2][W2][2*BASE_CH];
    static data_t reluB[H2][W2][2*BASE_CH];
    static data_t poolB[H4][W4][2*BASE_CH];
    static data_t upB[H2][W2][2*BASE_CH];

    conv2d_padded_golden<data_t, H2, W2, BASE_CH, 2*BASE_CH, K, PAD>(poolA, w2, b2, convB);
    batchnorm_golden<data_t, H2, W2, 2*BASE_CH>(convB, bn2_weight, bn2_bias, bn2_mean, bn2_var, eps, bnB);
    relu_golden<data_t, H2, W2, 2*BASE_CH>(bnB, reluB);
    maxpool_golden<data_t, H2, W2, 2*BASE_CH, 2>(reluB, poolB);
    nearest_neighbor_golden<data_t, H4, W4, 2*BASE_CH, 2>(poolB, upB);

    // ---- block C: conv -> BN+ReLU -> nearest upsample(2x) ----
    static data_t convC[H2][W2][BASE_CH];
    static data_t bnC[H2][W2][BASE_CH];
    static data_t reluC[H2][W2][BASE_CH];
    static data_t upC[H][W][BASE_CH];

    conv2d_padded_golden<data_t, H2, W2, 2*BASE_CH, BASE_CH, K, PAD>(upB, w3, b3, convC);
    batchnorm_golden<data_t, H2, W2, BASE_CH>(convC, bn3_weight, bn3_bias, bn3_mean, bn3_var, eps, bnC);
    relu_golden<data_t, H2, W2, BASE_CH>(bnC, reluC);
    nearest_neighbor_golden<data_t, H2, W2, BASE_CH, 2>(reluC, upC);

    // ---- block D: conv -> BN (no ReLU -- this is the output head) ----
    static data_t convD[H][W][OUT_CH];

    conv2d_padded_golden<data_t, H, W, BASE_CH, OUT_CH, K, PAD>(upC, w4, b4, convD);
    batchnorm_golden<data_t, H, W, OUT_CH>(convD, bn4_weight, bn4_bias, bn4_mean, bn4_var, eps, dout);
}
