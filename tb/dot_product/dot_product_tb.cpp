#include "dot_product.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    float a[N], b[N];
    float hw_result;
    double golden = 0.0;

    // Fixed seed -> identical stimulus every run, across every variant
    srand(42);

    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
        b[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        golden += static_cast<double>(a[i]) * static_cast<double>(b[i]); // double-precision reference
    }

    dot_product(a, b, &hw_result);

    double abs_err = std::fabs(hw_result - golden);
    double rel_err = abs_err / std::fabs(golden);

    printf("Golden (double):  %.8f\n", golden);
    printf("HW result:        %.8f\n", hw_result);
    printf("Abs error:        %.8e\n", abs_err);
    printf("Rel error:        %.8e\n", rel_err);

    // Tolerance accounts for float accumulation order differing
    // between naive, pipelined, and partial-sum variants
    const double REL_TOL = 1e-4;

    if (rel_err > REL_TOL) {
        printf("TEST FAILED\n");
        return 1;
    }

    printf("TEST PASSED\n");
    return 0;
}