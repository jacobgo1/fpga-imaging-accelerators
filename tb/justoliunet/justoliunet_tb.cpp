#include "justoliunet.hpp"
#include "justoliunet_vectors.hpp"

#include <cmath>
#include <cstdio>

// Compares the kernel with logits the exporter's numpy reference computed
// from the same checkpoint. float32 vs float64, so a small tolerance.
static bool close(float got, float want) {
    return std::fabs(got - want) <= 1e-4f + 1e-4f * std::fabs(want);
}

static int argmax(const float v[jl::CLASSES]) {
    int best = 0;
    for (int c = 1; c < jl::CLASSES; c++)
        if (v[c] > v[best]) best = c;
    return best;
}

int main() {
    int failures = 0;
    float worst = 0.0f;
    int per_class[jl::CLASSES] = {};
    for (int t = 0; t < JL_VECTORS; t++) {
        float logits[jl::CLASSES];
        justoliunet(jl_inputs[t], logits);
        bool ok = argmax(logits) == argmax(jl_expected[t]);
        for (int c = 0; c < jl::CLASSES; c++) {
            ok = ok && close(logits[c], jl_expected[t][c]);
            float err = std::fabs(logits[c] - jl_expected[t][c]);
            if (err > worst) worst = err;
        }
        per_class[argmax(logits)]++;
        if (!ok) {
            failures++;
            std::printf("FAIL pixel %d: got", t);
            for (int c = 0; c < jl::CLASSES; c++) std::printf(" %.6f", logits[c]);
            std::printf(", expected");
            for (int c = 0; c < jl::CLASSES; c++) std::printf(" %.6f", jl_expected[t][c]);
            std::printf("\n");
        }
    }
    std::printf("%s: %d of %d pixels match the reference (worst logit error %.2e); classes:",
                failures ? "FAIL" : "PASS", JL_VECTORS - failures, JL_VECTORS, worst);
    for (int c = 0; c < jl::CLASSES; c++) std::printf(" %d", per_class[c]);
    std::printf("\n");
    if (jl::PLACEHOLDER_NORMALIZATION)
        std::printf("NOTE: built with placeholder normalization (mean 0, std 1); "
                    "re-export with --mu-sd before judging real captures\n");
    return failures ? 1 : 0;
}
