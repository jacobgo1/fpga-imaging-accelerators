#pragma once

// One BatchNorm2d's parameters, in its state_dict's order (num_batches_tracked
// left out: inference does not use it). The blocks below hold their weights in
// structs like this one, so a model's whole state_dict maps onto one struct.
template<typename data_t, int CH>
struct batchnorm_params_golden {
    data_t weight[CH];
    data_t bias[CH];
    data_t running_mean[CH];
    data_t running_var[CH];
};
