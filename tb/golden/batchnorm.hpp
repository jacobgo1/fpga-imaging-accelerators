#pragma once

#include <cmath>
#include "../../src/golden/layers.hpp"

namespace golden {

// BatchNorm in inference mode (the running statistics), then A, in place on
// COUNT pixels or positions of CHANNELS values each. Only the float references
// have it: the kernels have it folded into their convs.
template<int CHANNELS, activation A = activation::none, typename T>
void batchnorm(T *data, int count,
               const T weight[CHANNELS], const T bias[CHANNELS],
               const T running_mean[CHANNELS], const T running_var[CHANNELS], T eps) {
    for (int i = 0; i < count; i++)
        for (int ch = 0; ch < CHANNELS; ch++) {
            T &x = data[i * CHANNELS + ch];
            const T normalized = (x - running_mean[ch]) / std::sqrt(running_var[ch] + eps);
            x = activate<A>(weight[ch] * normalized + bias[ch]);
        }
}

}  // namespace golden
