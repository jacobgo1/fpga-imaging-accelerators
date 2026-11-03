#ifndef JUSTOLIUNET_BN_HLS_HPP
#define JUSTOLIUNET_BN_HLS_HPP

#include "../../1D-Justo-LiuNet/justoliunet_golden/justoliunet.hpp"

// Golden 1D-Justo-LiuNet with BatchNorm: the same network as justoliunet, the
// BatchNorm folded into conv1, only the weights differ.

// din: [n][110] float. dout: [n][3] float (0 cloud, 1 land, 2 sea).
void justoliunet_bn(const data_t *din, data_t *dout, int n);

#endif
