#pragma once

// spectral_stem_golden.hpp on quantized weights, with BatchNorm folded into
// convsp1 (tools/quantize_weights.py): three 1x1 convs, each with a ReLU.
// Scales as in conv2d_same_quant_golden.hpp: out = sum(act * q) * w_scale + b_q * b_scale.
//
// The convs are 1x1, so each pixel is done on its own: its spectrum is read
// from din once, into a local array, and goes through all three layers. The
// patch is never stored whole (IN_CH bands of every pixel: 110 x 32 x 32
// floats), only the stem's output. din is the patch's H x W x IN_CH floats,
// HWC, starting at din[offset]: in a kernel, an AXI master pointer.
template<typename act_t, typename w_t, int IN_CH, int OUT_CH>
void pointwise_quant_relu(const act_t in[IN_CH],
                          const w_t weight[OUT_CH][IN_CH][1][1], act_t weight_scale,
                          const w_t bias[OUT_CH], act_t bias_scale,
                          act_t out[OUT_CH]) {
    for (int oc = 0; oc < OUT_CH; oc++) {
        act_t sum = 0;
        for (int ic = 0; ic < IN_CH; ic++)
            sum += in[ic] * (act_t)weight[oc][ic][0][0];
        act_t v = sum * weight_scale + (act_t)bias[oc] * bias_scale;
        out[oc] = v < 0 ? (act_t)0 : v;
    }
}

template<typename act_t, typename w_t, int H, int W, int IN_CH, int S1, int S2, int S3>
void spectral_stem_quant_golden(const act_t *din, long offset,
                                const w_t w1[S1][IN_CH][1][1], act_t w1_scale, const w_t b1[S1], act_t b1_scale,
                                const w_t w2[S2][S1][1][1],    act_t w2_scale, const w_t b2[S2], act_t b2_scale,
                                const w_t w3[S3][S2][1][1],    act_t w3_scale, const w_t b3[S3], act_t b3_scale,
                                act_t dout[H][W][S3]) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            act_t spectrum[IN_CH], a1[S1], a2[S2], a3[S3];
            const long pixel = offset + ((long)i * W + j) * IN_CH;
            for (int c = 0; c < IN_CH; c++)
                spectrum[c] = din[pixel + c];
            pointwise_quant_relu<act_t, w_t, IN_CH, S1>(spectrum, w1, w1_scale, b1, b1_scale, a1);
            pointwise_quant_relu<act_t, w_t, S1, S2>(a1, w2, w2_scale, b2, b2_scale, a2);
            pointwise_quant_relu<act_t, w_t, S2, S3>(a2, w3, w3_scale, b3, b3_scale, a3);
            for (int c = 0; c < S3; c++)
                dout[i][j][c] = a3[c];
        }
    }
}
