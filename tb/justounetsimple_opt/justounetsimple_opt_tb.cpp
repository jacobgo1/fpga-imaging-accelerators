// justounetsimple_opt_tb.cpp  --  testbench for the optimized kernel.
//
// The reference is the golden kernel (src/hls/justounetsimple/), run patch by
// patch on the same input. The optimized kernel gets all patches in one call,
// in its own input layout ([IN_CH_PAD/P1][H][W][P1] per patch, the real IC
// channels followed by IN_CH_PAD - IC zero channels). The sums are added in
// a different order, so the comparison has a small float tolerance.
#include "justounetsimple_opt.hpp"
#include "../../src/hls/justounetsimple/justounetsimple.cpp"
#include <cmath>
#include <cstdio>
#include <random>

constexpr int N = 2;  // patches; matches the m_axi depth pragmas
constexpr int H = JOPT_H, W = JOPT_W, IC = JOPT_IN_CH, ICP = JOPT_IN_CH_PAD;
constexpr int OC = JOPT_OUT_CH, P1 = JOPT_P1;
static_assert(H == JUNETS_H && W == JUNETS_W && IC == JUNETS_IN_CH && OC == JUNETS_OUT_CH,
              "the optimized and golden kernels must have the same geometry");

static float patches[N][H][W][IC];        // HWC, as the golden kernel takes them
static jopt_in_t din[N][ICP / P1][H][W];  // the optimized kernel's layout; zero-initialized
static float dout[N][H][W][OC];
static float expected[N][H][W][OC];

int main()
{
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);  // roughly z-scored bands
    for (int k = 0; k < N; k++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int c = 0; c < IC; c++) {  // channels [IC, ICP) stay zero: static init
                    float v = dist(rng);
                    patches[k][y][x][c] = v;
                    din[k][c / P1][y][x][c % P1] = v;
                }

    for (int k = 0; k < N; k++)
        justounetsimple(patches[k], expected[k]);

    justounetsimple_opt(&din[0][0][0][0], &dout[0][0][0][0], N);

    int errors = 0;
    float worst = 0;
    const float *a = &dout[0][0][0][0], *e = &expected[0][0][0][0];
    constexpr int count = N * H * W * OC;
    for (int i = 0; i < count; i++) {
        float diff = std::fabs(a[i] - e[i]);
        if (diff > worst) worst = diff;
        if (diff > 1e-3f * (1.0f + std::fabs(e[i]))) {
            if (errors < 5)
                printf("mismatch at [%d]: got %g, expected %g\n", i, (double)a[i], (double)e[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("PASS  (%d patches, %d outputs, worst difference %g)\n", N, count, (double)worst);
    else
        printf("FAIL  (%d mismatches)\n", errors);
    return errors == 0 ? 0 : 1;
}
