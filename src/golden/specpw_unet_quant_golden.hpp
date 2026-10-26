#pragma once

#include "quant_scale.hpp"
#include "specpw_unet_golden.hpp"   // the sizes: specpw_unet_small, _medium, _large
#include "spectral_stem_quant_golden.hpp"
#include "conv_block_quant_golden.hpp"
#include "up_block_quant_golden.hpp"
#include "conv2d_same_quant_golden.hpp"
#include "maxpool_golden.hpp"
#include "concat_golden.hpp"

// The quantized SpecPW-UNet: specpw_unet_golden.hpp with every BatchNorm folded
// into the conv before it and int8 weights (tools/quantize_weights.py), so
// every layer is a conv with a bias, then a ReLU (not the final 1x1):
//
//   x    = stem(x)                                  three 1x1 convs
//   skip = double1(x)
//   x    = maxpool(double2(maxpool(skip)))
//   x    = up2(double3(up1(x)))
//   x    = double4(cat([x, skip]))
//   out  = final(x)
//
// Each weight w comes with its scale w_s = 2^-frac (conv2d_same_quant_golden.hpp);
// the arguments are in the state_dict's order. din is one H x W x IN_CH patch,
// HWC, at din[offset] (the stem reads it pixel by pixel; in a kernel, straight
// from DDR). H and W must be divisible by 4; the models were trained on 32 x 32.
//
// The maps that are 32 x 32 x 32 floats or more in the large model are
// UltraRAM, so that BRAM holds the weights (622 KB of float ROM in the large
// model). Small maps stay in BRAM.
template<typename act_t, typename w_t, int H, int W, int IN_CH, int N_CLS, typename Size>
void specpw_unet_quant_golden(
    const act_t *din, long offset,
    const w_t sp1_w[Size::S1][IN_CH][1][1],     act_t sp1_ws, const w_t sp1_b[Size::S1], act_t sp1_bs,
    const w_t sp2_w[Size::S2][Size::S1][1][1],  act_t sp2_ws, const w_t sp2_b[Size::S2], act_t sp2_bs,
    const w_t sp3_w[Size::S3][Size::S2][1][1],  act_t sp3_ws, const w_t sp3_b[Size::S3], act_t sp3_bs,
    const w_t d1a_w[Size::C1][Size::S3][3][3],  act_t d1a_ws, const w_t d1a_b[Size::C1], act_t d1a_bs,
    const w_t d1b_w[Size::C1][Size::C1][3][3],  act_t d1b_ws, const w_t d1b_b[Size::C1], act_t d1b_bs,
    const w_t d2a_w[Size::C2][Size::C1][3][3],  act_t d2a_ws, const w_t d2a_b[Size::C2], act_t d2a_bs,
    const w_t d2b_w[Size::C2][Size::C2][3][3],  act_t d2b_ws, const w_t d2b_b[Size::C2], act_t d2b_bs,
    const w_t up1_w[Size::C2][Size::C2][3][3],  act_t up1_ws, const w_t up1_b[Size::C2], act_t up1_bs,
    const w_t d3a_w[Size::C3][Size::C2][3][3],  act_t d3a_ws, const w_t d3a_b[Size::C3], act_t d3a_bs,
    const w_t d3b_w[Size::C3][Size::C3][3][3],  act_t d3b_ws, const w_t d3b_b[Size::C3], act_t d3b_bs,
    const w_t up2_w[Size::C3][Size::C3][3][3],  act_t up2_ws, const w_t up2_b[Size::C3], act_t up2_bs,
    const w_t d4a_w[Size::C4][Size::C3 + Size::C1][3][3], act_t d4a_ws, const w_t d4a_b[Size::C4], act_t d4a_bs,
    const w_t d4b_w[Size::C4][Size::C4][3][3],  act_t d4b_ws, const w_t d4b_b[Size::C4], act_t d4b_bs,
    const w_t fin_w[N_CLS][Size::C4][1][1],     act_t fin_ws, const w_t fin_b[N_CLS],    act_t fin_bs,
    act_t dout[H][W][N_CLS])
{
    static_assert(H % 4 == 0 && W % 4 == 0, "H and W must be divisible by 4 (two maxpools)");
    constexpr int S3 = Size::S3, C1 = Size::C1, C2 = Size::C2, C3 = Size::C3, C4 = Size::C4;
    constexpr int H2 = H / 2, W2 = W / 2, H4 = H / 4, W4 = W / 4;

    static act_t stem[H][W][S3];
    static act_t skip[H][W][C1];
    #pragma HLS BIND_STORAGE variable=skip type=ram_2p impl=uram
    static act_t pool1[H2][W2][C1];
    static act_t enc2[H2][W2][C2];
    #pragma HLS BIND_STORAGE variable=enc2 type=ram_2p impl=uram
    static act_t pool2[H4][W4][C2];
    static act_t up1[H2][W2][C2];
    #pragma HLS BIND_STORAGE variable=up1 type=ram_2p impl=uram
    static act_t dec3[H2][W2][C3];
    static act_t up2[H][W][C3];
    #pragma HLS BIND_STORAGE variable=up2 type=ram_2p impl=uram
    static act_t cat[H][W][C3 + C1];
    #pragma HLS BIND_STORAGE variable=cat type=ram_2p impl=uram
    static act_t dec4[H][W][C4];
    #pragma HLS BIND_STORAGE variable=dec4 type=ram_2p impl=uram

    spectral_stem_quant_golden<act_t, w_t, H, W, IN_CH, Size::S1, Size::S2, S3>(
        din, offset, sp1_w, sp1_ws, sp1_b, sp1_bs, sp2_w, sp2_ws, sp2_b, sp2_bs,
        sp3_w, sp3_ws, sp3_b, sp3_bs, stem);

    // encoder
    conv_block_quant_golden<act_t, w_t, H, W, S3, C1>(
        stem, d1a_w, d1a_ws, d1a_b, d1a_bs, d1b_w, d1b_ws, d1b_b, d1b_bs, skip);
    maxpool_golden<act_t, H, W, C1, 2>(skip, pool1);
    conv_block_quant_golden<act_t, w_t, H2, W2, C1, C2>(
        pool1, d2a_w, d2a_ws, d2a_b, d2a_bs, d2b_w, d2b_ws, d2b_b, d2b_bs, enc2);
    maxpool_golden<act_t, H2, W2, C2, 2>(enc2, pool2);

    // decoder
    up_block_quant_golden<act_t, w_t, H4, W4, C2, C2>(pool2, up1_w, up1_ws, up1_b, up1_bs, up1);
    conv_block_quant_golden<act_t, w_t, H2, W2, C2, C3>(
        up1, d3a_w, d3a_ws, d3a_b, d3a_bs, d3b_w, d3b_ws, d3b_b, d3b_bs, dec3);
    up_block_quant_golden<act_t, w_t, H2, W2, C3, C3>(dec3, up2_w, up2_ws, up2_b, up2_bs, up2);

    // skip connection: the upsampled map first, then the encoder's
    concat_golden<act_t, H, W, C3, C1>(up2, skip, cat);
    conv_block_quant_golden<act_t, w_t, H, W, C3 + C1, C4>(
        cat, d4a_w, d4a_ws, d4a_b, d4a_bs, d4b_w, d4b_ws, d4b_b, d4b_bs, dec4);

    conv2d_same_quant_golden<act_t, w_t, H, W, C4, N_CLS, 1>(dec4, fin_w, fin_ws, fin_b, fin_bs, dout);
}
