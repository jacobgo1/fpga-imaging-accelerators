#ifndef SP_UNET_OPT_ZYNQ7000_HLS_HPP
#define SP_UNET_OPT_ZYNQ7000_HLS_HPP

#include <stdint.h>
#include "../../../../common/hls_vector_shim.hpp"
#include "../../../../common/hls_burst_maxi_shim.hpp"

// The direct SpecPW-UNet kernels sized for a Zynq-7000: the 7020 (board pynqz2
// or zedboard, at most 205 of its 220 DSPs) and the 7030 (board picobob, 400).
// LUTs and flip-flops are the scarce part, so every conv keeps its window in three
// row memories (conv3x3_tiled_ram) and sends ReLU'd activations itself; the stem's
// later layers take a few inputs a step on a few DSPs; one or two 64-bit read ports
// (each kernel's READ_PORTS); the FIFOs between the processes 2 deep (16 would save
// 7 % of the time on paper, 0.13 % measured, and the LUTs are not there); the skip in
// block RAM. See optimizations_implemented/sp_unet.md.

constexpr int PATCH_SIZE = 32;
constexpr int CUBE_BANDS = 120;
constexpr int MODEL_BANDS = 110;                // raw bands 8-117
constexpr int CLASSES = 3;
constexpr int BANDS_PER_WORD = 4;               // one 64-bit word

constexpr int ZSCORE_FRAC_BITS = 5;             // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 10;        // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 16;             // ... of the scores

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// A conv's size in a kernel's table: input and output channels and taps of the
// window a cycle (1, or a column of 3), and whether its multipliers are DSPs
// (packed convs: always).
template<int IN_PER_CYCLE_, int OUT_PER_CYCLE_, int TAPS_PER_STEP_, bool DSP_>
struct tiled_ram_layer_size {
    static constexpr int IN_PER_CYCLE = IN_PER_CYCLE_, OUT_PER_CYCLE = OUT_PER_CYCLE_;
    static constexpr int TAPS_PER_STEP = TAPS_PER_STEP_;
    static constexpr bool DSP = DSP_;
};

#endif
