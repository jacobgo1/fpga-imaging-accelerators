#pragma once

// One BatchNorm2d's parameters, in its state_dict's order.
template<typename data_t, int CHANNELS>
struct batchnorm_params_golden {
    data_t weight[CHANNELS];
    data_t bias[CHANNELS];
    data_t running_mean[CHANNELS];
    data_t running_var[CHANNELS];
};
