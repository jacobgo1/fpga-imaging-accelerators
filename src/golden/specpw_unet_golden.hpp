#pragma once

#include "spectral_stem_golden.hpp"
#include "conv_block_golden.hpp"
#include "up_block_golden.hpp"
#include "maxpool_golden.hpp"
#include "concat_golden.hpp"
#include "conv2d_golden.hpp"

// SpecPW-UNet (the sp_unet_small/medium/large checkpoints) as trained, in float,
// from the plain modules. A spectral stem of 1x1 convs, then a two-level U-Net
// with one skip connection:
//
//   x    = stem(x)                                  IN_CH -> S1 -> S2 -> S3
//   skip = double1(x)                               S3 -> C1          H x W
//   x    = maxpool(double2(maxpool(skip)))          C1 -> C2          H/4 x W/4
//   x    = double3(up1(x))                          C2 -> C2 -> C3    H/2 x W/2
//   x    = up2(x)                                   C3 -> C3          H x W
//   x    = double4(cat([x, skip]))                  C3 + C1 -> C4
//   out  = final(x)                                 C4 -> N_CLS, 1x1 conv with bias
//
// The sizes (the figure's channel counts; the checkpoints' tensor shapes):
//
//            stem S1 S2 S3   C1  C2  C3  C4
//   small        24 12 12    12  24  12   8
//   medium       24 16 12    24  48  24  12
//   large        32 16 16    32  64  32  16
//
// The weights struct holds every tensor of the state_dict in its order
// (num_batches_tracked left out), each in PyTorch's own layout, so a checkpoint
// loads as one block of floats. Activations are HWC. H and W must be divisible
// by 4; the models were trained on 32 x 32 patches. eps is BatchNorm2d's (1e-5).
struct specpw_unet_small  { static constexpr int S1 = 24, S2 = 12, S3 = 12, C1 = 12, C2 = 24, C3 = 12, C4 = 8; };
struct specpw_unet_medium { static constexpr int S1 = 24, S2 = 16, S3 = 12, C1 = 24, C2 = 48, C3 = 24, C4 = 12; };
struct specpw_unet_large  { static constexpr int S1 = 32, S2 = 16, S3 = 16, C1 = 32, C2 = 64, C3 = 32, C4 = 16; };

template<typename data_t, int IN_CH, int N_CLS, typename Size>
struct specpw_unet_params_golden {
    spectral_stem_params_golden<data_t, IN_CH, Size::S1, Size::S2, Size::S3> stem;
    conv_block_params_golden<data_t, Size::S3, Size::C1> double1;
    conv_block_params_golden<data_t, Size::C1, Size::C2> double2;
    up_block_params_golden<data_t, Size::C2, Size::C2> up1;
    conv_block_params_golden<data_t, Size::C2, Size::C3> double3;
    up_block_params_golden<data_t, Size::C3, Size::C3> up2;
    conv_block_params_golden<data_t, Size::C3 + Size::C1, Size::C4> double4;
    data_t final_w[N_CLS][Size::C4][1][1];
    data_t final_b[N_CLS];
};

template<typename data_t, int H, int W, int IN_CH, int N_CLS, typename Size>
void specpw_unet_golden(const data_t din[H][W][IN_CH],
                        const specpw_unet_params_golden<data_t, IN_CH, N_CLS, Size> &p,
                        data_t eps,
                        data_t dout[H][W][N_CLS]) {
    static_assert(H % 4 == 0 && W % 4 == 0, "H and W must be divisible by 4 (two maxpools)");
    constexpr int S3 = Size::S3, C1 = Size::C1, C2 = Size::C2, C3 = Size::C3, C4 = Size::C4;
    constexpr int H2 = H / 2, W2 = W / 2, H4 = H / 4, W4 = W / 4;

    static data_t stem[H][W][S3];
    static data_t skip[H][W][C1];
    static data_t pool1[H2][W2][C1];
    static data_t enc2[H2][W2][C2];
    static data_t pool2[H4][W4][C2];
    static data_t up1[H2][W2][C2];
    static data_t dec3[H2][W2][C3];
    static data_t up2[H][W][C3];
    static data_t cat[H][W][C3 + C1];
    static data_t dec4[H][W][C4];

    spectral_stem_golden<data_t, H, W, IN_CH, Size::S1, Size::S2, S3>(din, p.stem, eps, stem);

    // encoder
    conv_block_golden<data_t, H, W, S3, C1>(stem, p.double1, eps, skip);
    maxpool_golden<data_t, H, W, C1, 2>(skip, pool1);
    conv_block_golden<data_t, H2, W2, C1, C2>(pool1, p.double2, eps, enc2);
    maxpool_golden<data_t, H2, W2, C2, 2>(enc2, pool2);

    // decoder
    up_block_golden<data_t, H4, W4, C2, C2>(pool2, p.up1, eps, up1);
    conv_block_golden<data_t, H2, W2, C2, C3>(up1, p.double3, eps, dec3);
    up_block_golden<data_t, H2, W2, C3, C3>(dec3, p.up2, eps, up2);

    // skip connection: the upsampled map first, then the encoder's
    concat_golden<data_t, H, W, C3, C1>(up2, skip, cat);
    conv_block_golden<data_t, H, W, C3 + C1, C4>(cat, p.double4, eps, dec4);

    conv2d_golden<data_t, H, W, C4, N_CLS, 1>(dec4, p.final_w, p.final_b, dout);
}
