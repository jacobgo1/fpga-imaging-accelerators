#include "justoliunet.hpp"

namespace {

// One network stage: valid 1D convolution, ReLU, then max-pool by 2
// (dropping an odd last sample, as PyTorch's MaxPool1d does). ReLU and max
// commute, so pooling starts from 0 and applies both at once.
template <int IN_CH, int IN_LEN, int OUT_CH>
void conv_relu_pool(const float in[IN_CH][IN_LEN],
                    const float weights[OUT_CH][IN_CH][jl::K],
                    const float bias[OUT_CH],
                    float out[OUT_CH][(IN_LEN - jl::K + 1) / 2]) {
    constexpr int OUT_LEN = (IN_LEN - jl::K + 1) / 2;
    out_channel:
    for (int o = 0; o < OUT_CH; o++) {
        out_position:
        for (int p = 0; p < OUT_LEN; p++) {
            float pooled = 0.0f;
            pool_pair:
            for (int s = 0; s < 2; s++) {
                const int start = 2 * p + s;
                float acc = bias[o];
                in_channel:
                for (int i = 0; i < IN_CH; i++) {
                    tap:
                    for (int k = 0; k < jl::K; k++)
                        acc += in[i][start + k] * weights[o][i][k];
                }
                if (acc > pooled) pooled = acc;
            }
            out[o][p] = pooled;
        }
    }
}

}  // namespace

void justoliunet(const float spectrum[jl::BANDS], float logits[jl::CLASSES]) {
    // Spectrum in and logits out are AXI-Lite registers, like matmul, so the
    // PS can drive the kernel over JTAG without a DMA.
    #pragma HLS INTERFACE mode=s_axilite port=spectrum bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=logits bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    float x0[1][jl::BANDS];
    float x1[jl::C1][jl::L1];
    float x2[jl::C2][jl::L2];
    float x3[jl::C3][jl::L3];
    float x4[jl::C4][jl::L4];

    load_spectrum:
    for (int b = 0; b < jl::BANDS; b++) x0[0][b] = spectrum[b];

    conv_relu_pool<1, jl::BANDS, jl::C1>(x0, jl::conv1_w, jl::conv1_b, x1);
    conv_relu_pool<jl::C1, jl::L1, jl::C2>(x1, jl::conv2_w, jl::conv2_b, x2);
    conv_relu_pool<jl::C2, jl::L2, jl::C3>(x2, jl::conv3_w, jl::conv3_b, x3);
    conv_relu_pool<jl::C3, jl::L3, jl::C4>(x3, jl::conv4_w, jl::conv4_b, x4);

    dense:
    for (int c = 0; c < jl::CLASSES; c++) {
        float acc = jl::fc_b[c];
        for (int ch = 0; ch < jl::C4; ch++)
            for (int l = 0; l < jl::L4; l++)
                acc += x4[ch][l] * jl::fc_w[c][ch * jl::L4 + l];
        logits[c] = acc;
    }
}
