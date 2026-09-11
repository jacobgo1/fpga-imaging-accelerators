#include "matmul.hpp"
#include <iostream>

int main() {
    data_t a[MAT_SIZE][MAT_SIZE], b[MAT_SIZE][MAT_SIZE];
    result_t actual[MAT_SIZE][MAT_SIZE];
    unsigned state = 42;
    for (int test = 0; test < 24; ++test) {
        for (int i = 0; i < MAT_SIZE; ++i) {
            for (int j = 0; j < MAT_SIZE; ++j) {
                state = state * 1664525u + 1013904223u;
                a[i][j] = static_cast<data_t>(static_cast<int>(state % 33) - 16);
                state = state * 1664525u + 1013904223u;
                b[i][j] = static_cast<data_t>(static_cast<int>(state % 33) - 16);
                if (test == 0) a[i][j] = 0;
                if (test == 1) b[i][j] = (i == j) ? 1 : 0;
                if (test == 2) { a[i][j] = -32768; b[i][j] = 32767; }
                actual[i][j] = 12345;
            }
        }
        matmul(a, b, actual);
        for (int i = 0; i < MAT_SIZE; ++i) {
            for (int j = 0; j < MAT_SIZE; ++j) {
                long long expected = 0;
                for (int k = 0; k < MAT_SIZE; ++k)
                    expected += static_cast<long long>(a[i][k]) * b[k][j];
                if (actual[i][j] != expected) {
                    std::cerr << "Mismatch: case " << test << " at " << i << ',' << j
                              << ": " << actual[i][j] << " != " << expected << '\n';
                    return 1;
                }
            }
        }
    }
    std::cout << "PASS: 24 matrix tests (zero, identity, extremes, signed random, repeated calls)\n";
    return 0;
}
