#ifndef SP_UNET_OPT_HLS_HPP
#define SP_UNET_OPT_HLS_HPP

#include <stdint.h>
#include "../../common/hls_vector_shim.hpp"
#include "../../common/hls_burst_maxi_shim.hpp"

// The optimized SpecPW-UNet (small, medium and large) on a whole raw capture,
// for the ZCU104. The 3x3 convs are integer Winograd F(2x2, 3x3) and the
// UpBlocks are computed from the low-res maps (wino_conv3x3.hpp); the maps
// between them travel in 2x2-block order. The skip connection waits in a FIFO
// in UltraRAM. Every process is sized for the same time a patch, in each
// kernel's .cpp. Bit for bit classify_fixed in tools/specpw_unet_model.py.
// Design and numbers: optimizations_implemented/sp_unet.md.

constexpr int PATCH_SIZE = 32;
constexpr int CUBE_BANDS = 120;
constexpr int MODEL_BANDS = 110;                // raw bands 8-117
constexpr int CLASSES = 3;
constexpr int READ_PORTS = 4;
constexpr int BANDS_PER_WORD = 8;               // one 128-bit word

constexpr int ZSCORE_FRAC_BITS = 5;             // fraction bits of the int8 z-scores
constexpr int ACTIVATION_FRAC_BITS = 10;        // ... of the int16 activations
constexpr int SCORE_FRAC_BITS = 16;             // ... of the scores

typedef hls::vector<uint16_t, BANDS_PER_WORD> cube_word_t;
typedef hls::vector<int32_t, 4> score_word_t;   // 3 scores and the class

// A layer's size in a kernel's table: input and output channels a cycle, and
// whether its multipliers are DSPs or LUTs.
template<int IN_PER_CYCLE_, int OUT_PER_CYCLE_, bool DSP_>
struct layer_size {
    static constexpr int IN_PER_CYCLE = IN_PER_CYCLE_, OUT_PER_CYCLE = OUT_PER_CYCLE_;
    static constexpr bool DSP = DSP_;
};

#endif
