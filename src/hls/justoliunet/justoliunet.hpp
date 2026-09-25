#ifndef JUSTOLIUNET_HPP
#define JUSTOLIUNET_HPP

// 1D-Justo-LiuNet (Justo et al., arXiv:2310.16210): classifies one pixel as
// one of CLASSES from its min-max normalized spectrum.
//
//   spectrum [BANDS]
//   4 x [ Conv1D kernel K, valid -> ReLU -> MaxPool 2 ]
//   flatten (channel-major, as PyTorch) -> Dense -> logits [CLASSES]
//
// Sizes and weights come from justoliunet_weights.hpp, generated from a
// trained checkpoint by tools/export_justoliunet.py. float32 throughout, so
// results match the trained model; fixed point is the next optimization.

#include "justoliunet_weights.hpp"

namespace jl {
constexpr int L1 = (BANDS - K + 1) / 2;  // length after each conv + pool
constexpr int L2 = (L1 - K + 1) / 2;
constexpr int L3 = (L2 - K + 1) / 2;
constexpr int L4 = (L3 - K + 1) / 2;
constexpr int FEATURES = C4 * L4;
static_assert(L4 >= 1, "spectrum too short for four conv + pool stages");
}

void justoliunet(const float spectrum[jl::BANDS], float logits[jl::CLASSES]);

#endif
