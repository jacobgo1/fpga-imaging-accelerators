// justoliunet_bn_tb.cpp  --  testbench for the quantized 1D-Justo-LiuNet with
// BatchNorm. The kernel checks are justoliunet's (justoliunet_check.hpp); on
// top, the BatchNorm itself: the float network with a BatchNorm after conv1
// must equal the same network with that BatchNorm folded into conv1, which is
// what tools/quantize_weights.py does to the trained model and why the kernel
// has no BatchNorm of its own.
#include "justoliunet_bn.hpp"
#include "../../weights/quantized/justoliunet_bn/justoliunet_bn_int8.hpp"

namespace q = justoliunet_bn_int8;

static void run_kernel(const float *din, float *dout, int n) {
    justoliunet_bn(din, dout, n);
}

#include "../justoliunet/justoliunet_check.hpp"

// BatchNorm with random parameters, against it folded into conv1:
// scale = gamma / sqrt(var + eps), w1' = w1 * scale, b1' = (b1 - mean) * scale + beta.
static int check_batchnorm()
{
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> any(-1.0f, 1.0f), positive(0.05f, 2.0f);
    float g[B], beta[B], mean[B], var[B], fw1[B][1][K], fb1[B];
    const float eps = 1e-5f;
    for (int c = 0; c < B; c++) {
        g[c] = positive(rng); beta[c] = any(rng); mean[c] = any(rng); var[c] = positive(rng);
        const float scale = g[c] / std::sqrt(var[c] + eps);
        for (int k = 0; k < K; k++) fw1[c][0][k] = w1[c][0][k] * scale;
        fb1[c] = (b1[c] - mean[c]) * scale + beta[c];
    }
    int errors = 0;
    for (int p = 0; p < PIXELS; p++) {
        float with_bn[C], folded[C];
        reference(din[p], w1, b1, g, beta, mean, var, eps, with_bn);
        reference(din[p], fw1, fb1, bn_one, bn_zero, bn_zero, bn_one, 0.0f, folded);
        for (int c = 0; c < C; c++)
            if (std::fabs(with_bn[c] - folded[c]) > 1e-4f * (1.0f + std::fabs(folded[c]))) {
                if (errors < 5)
                    printf("BatchNorm: pixel %d class %d: %g with it, %g folded\n", p, c,
                           (double)with_bn[c], (double)folded[c]);
                errors++;
            }
    }
    printf("%s  (BatchNorm after conv1 = BatchNorm folded into conv1, %d pixels)\n",
           errors ? "FAIL" : "PASS", PIXELS);
    return errors == 0 ? 0 : 1;
}

int main()
{
    const int kernel = check_kernel();   // also loads the weights and the spectra
    const int bn = check_batchnorm();
    return kernel || bn;
}
