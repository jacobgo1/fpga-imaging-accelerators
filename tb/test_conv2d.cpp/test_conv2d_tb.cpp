#include "conv2d.hpp"
#include <iostream>

int main() {
    data_t in[IN_CH][IN_H][IN_W];
    data_t weights[OUT_CH][IN_CH][KERNEL_SIZE][KERNEL_SIZE];
    result_t actual[OUT_CH][OUT_H][OUT_W];
    unsigned state = 42;
    for (int test = 0; test < 8; ++test) {
        for (int ic = 0; ic < IN_CH; ++ic) {
            for (int i = 0; i < IN_H; ++i) {
                for (int j = 0; j < IN_W; ++j) {
                    state = state * 1664525u + 1013904223u;
                    in[ic][i][j] = static_cast<data_t>(static_cast<int>(state % 33) - 16);
                    if (test == 0) in[ic][i][j] = 0;
                    if (test == 1) in[ic][i][j] = (i == j) ? 1 : 0;
                    if (test == 2) in[ic][i][j] = -32768;
                }
            }
        }
        for (int oc = 0; oc < OUT_CH; ++oc) {
            for (int ic = 0; ic < IN_CH; ++ic) {
                for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                    for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                        state = state * 1664525u + 1013904223u;
                        weights[oc][ic][kh][kw] =
                            static_cast<data_t>(static_cast<int>(state % 33) - 16);
                        if (test == 2) weights[oc][ic][kh][kw] = 32767;
                    }
                }
            }
        }
        for (int oc = 0; oc < OUT_CH; ++oc)
            for (int i = 0; i < OUT_H; ++i)
                for (int j = 0; j < OUT_W; ++j)
                    actual[oc][i][j] = 12345;

        conv2d(in, weights, actual);

        for (int oc = 0; oc < OUT_CH; ++oc) {
            for (int oh = 0; oh < OUT_H; ++oh) {
                for (int ow = 0; ow < OUT_W; ++ow) {
                    long long expected = 0;
                    for (int ic = 0; ic < IN_CH; ++ic)
                        for (int kh = 0; kh < KERNEL_SIZE; ++kh)
                            for (int kw = 0; kw < KERNEL_SIZE; ++kw)
                                expected += static_cast<long long>(in[ic][oh + kh][ow + kw]) *
                                            weights[oc][ic][kh][kw];
                    if (actual[oc][oh][ow] != expected) {
                        std::cerr << "Mismatch: case " << test << " at oc=" << oc << " "
                                  << oh << ',' << ow << ": " << actual[oc][oh][ow]
                                  << " != " << expected << '\n';
                        return 1;
                    }
                }
            }
        }
    }
    std::cout << "PASS: 8 conv2d tests (zero, identity, extremes, signed random, repeated calls)\n";
    return 0;
}
