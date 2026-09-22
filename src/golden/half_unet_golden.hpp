#pragma once

#include "doubleconv_golden.hpp"
#include "maxpool_golden.hpp"
#include "nearest_neighbor_golden.hpp"
#include "add_golden.hpp"
#include "conv1x1_golden.hpp"

// A "half" U-Net: an ordinary contracting encoder (doubleconv + maxpool,
// four times), but no matching decoder of successive up-convs. Instead,
// every encoder stage's output is nearest-neighbor upsampled straight to
// full resolution and fused into the finest-resolution encoder output by
// plain addition, not concatenation -- so the channel count stays IM_CH
// throughout and no extra fuse weights are needed. One more doubleconv and
// a 1x1 head produce the final OUT_CH-channel map.
//
// Every weight/bias pair matches PyTorch's native Conv2d parameters
// ([out][in][kh][kw] + [out]) so a real trained model's state_dict can be
// loaded directly with no reshuffling. The one thing PyTorch and this file
// disagree on is which axis channels sit on for the *activations*: PyTorch
// tensors are NCHW (channels first), every array here is HWC (channels
// last). Transpose input/output tensors, not weights, when comparing
// against a PyTorch reference -- e.g. `tensor.permute(1, 2, 0).numpy()`.
//
// H and W must be divisible by 16 (four halvings): sizes that aren't will
// fail to compile, not silently misbehave, because the upsampled skip
// arrays are declared at exactly [H][W][...] and an inexact
// (H/16)*16 != H mismatches that array type.
//
// Intermediate feature maps are "static": at realistic image sizes they
// are several megabytes each, too big to put on the call stack safely.
template<typename data_t, int H, int W, int IN_CH, int IM_CH, int OUT_CH, int K, int POOL>
void half_unet_golden(
    data_t din[H][W][IN_CH],
    data_t w1[IM_CH][IN_CH][K][K],  data_t b1[IM_CH],
    data_t w2[IM_CH][IM_CH][K][K],  data_t b2[IM_CH],
    data_t w3[IM_CH][IM_CH][K][K],  data_t b3[IM_CH],
    data_t w4[IM_CH][IM_CH][K][K],  data_t b4[IM_CH],
    data_t w5[IM_CH][IM_CH][K][K],  data_t b5[IM_CH],
    data_t w6[IM_CH][IM_CH][K][K],  data_t b6[IM_CH],
    data_t w7[IM_CH][IM_CH][K][K],  data_t b7[IM_CH],
    data_t w8[IM_CH][IM_CH][K][K],  data_t b8[IM_CH],
    data_t w9[IM_CH][IM_CH][K][K],  data_t b9[IM_CH],
    data_t w10[IM_CH][IM_CH][K][K], data_t b10[IM_CH],
    data_t w11[IM_CH][IM_CH][K][K], data_t b11[IM_CH],
    data_t w12[IM_CH][IM_CH][K][K], data_t b12[IM_CH],
    data_t w13[OUT_CH][IM_CH],      data_t b13[OUT_CH],
    data_t dout[H][W][OUT_CH])
{
    static data_t doubleconv1[H][W][IM_CH];
    static data_t maxpool1[H/2][W/2][IM_CH];
    static data_t doubleconv2[H/2][W/2][IM_CH];
    static data_t maxpool2[H/4][W/4][IM_CH];
    static data_t doubleconv3[H/4][W/4][IM_CH];
    static data_t maxpool3[H/8][W/8][IM_CH];
    static data_t doubleconv4[H/8][W/8][IM_CH];
    static data_t maxpool4[H/16][W/16][IM_CH];
    static data_t doubleconv5[H/16][W/16][IM_CH];

    static data_t skip1[H][W][IM_CH];
    static data_t skip2[H][W][IM_CH];
    static data_t skip3[H][W][IM_CH];
    static data_t skip4[H][W][IM_CH];
    static data_t fused1[H][W][IM_CH];
    static data_t fused2[H][W][IM_CH];
    static data_t fused3[H][W][IM_CH];
    static data_t fused4[H][W][IM_CH];
    static data_t doubleconv6[H][W][IM_CH];

    // ---- encoder ----
    doubleconv_golden<data_t, H, W, IN_CH, IM_CH, IM_CH, K>(din, w1, b1, w2, b2, doubleconv1);
    maxpool_golden<data_t, H, W, IM_CH, POOL>(doubleconv1, maxpool1);

    doubleconv_golden<data_t, H/2, W/2, IM_CH, IM_CH, IM_CH, K>(maxpool1, w3, b3, w4, b4, doubleconv2);
    maxpool_golden<data_t, H/2, W/2, IM_CH, POOL>(doubleconv2, maxpool2);

    doubleconv_golden<data_t, H/4, W/4, IM_CH, IM_CH, IM_CH, K>(maxpool2, w5, b5, w6, b6, doubleconv3);
    maxpool_golden<data_t, H/4, W/4, IM_CH, POOL>(doubleconv3, maxpool3);

    doubleconv_golden<data_t, H/8, W/8, IM_CH, IM_CH, IM_CH, K>(maxpool3, w7, b7, w8, b8, doubleconv4);
    maxpool_golden<data_t, H/8, W/8, IM_CH, POOL>(doubleconv4, maxpool4);

    doubleconv_golden<data_t, H/16, W/16, IM_CH, IM_CH, IM_CH, K>(maxpool4, w9, b9, w10, b10, doubleconv5);

    // ---- multi-scale upsample ----
    nearest_neighbor_golden<data_t, H/2,  W/2,  IM_CH, 2>(doubleconv2, skip1);
    nearest_neighbor_golden<data_t, H/4,  W/4,  IM_CH, 4>(doubleconv3, skip2);
    nearest_neighbor_golden<data_t, H/8,  W/8,  IM_CH, 8>(doubleconv4, skip3);
    nearest_neighbor_golden<data_t, H/16, W/16, IM_CH, 16>(doubleconv5, skip4);

    // ---- fuse: doubleconv1 + skip1 + skip2 + skip3 + skip4 ----
    add_golden<data_t, H, W, IM_CH>(doubleconv1, skip1, fused1);
    add_golden<data_t, H, W, IM_CH>(fused1, skip2, fused2);
    add_golden<data_t, H, W, IM_CH>(fused2, skip3, fused3);
    add_golden<data_t, H, W, IM_CH>(fused3, skip4, fused4);

    // ---- head ----
    doubleconv_golden<data_t, H, W, IM_CH, IM_CH, IM_CH, K>(fused4, w11, b11, w12, b12, doubleconv6);
    conv1x1_golden<data_t, H, W, IM_CH, OUT_CH>(doubleconv6, w13, b13, dout);
}
