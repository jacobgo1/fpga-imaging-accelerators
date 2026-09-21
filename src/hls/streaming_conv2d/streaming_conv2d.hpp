#ifndef STREAMING_CONV2D_HPP
#define STREAMING_CONV2D_HPP

#include <cstdint>

// =====================================================================
// 1. Data types
// =====================================================================
using data_t   = std::int16_t;   // input pixels and weights
using result_t = std::int64_t;   // accumulator and output

// =====================================================================
// 2. Layer geometry  --  the only block you normally edit
//
//    Prefixed SCONV_ (not CONV_, which conv2d.hpp already uses) so this
//    kernel can be compiled alongside conv2d without its macros colliding
//    -- macros are global text substitution with no namespace of their own.
// =====================================================================
#define SCONV_IN_CHANNELS       3   // input channels (e.g. spectral bands)
#define SCONV_OUT_CHANNELS      3   // output channels (learned feature maps)
#define SCONV_IN_HEIGHT        28   // input rows -- can be large; see part 6
#define SCONV_IN_WIDTH         28   // input columns -- can be large; see part 6
#define SCONV_KERNEL_HEIGHT     3   // convolution window rows
#define SCONV_KERNEL_WIDTH      3   // convolution window columns
#define SCONV_STRIDE             1   // window step per output pixel; no padding ("valid")

#define SCONV_OUT_HEIGHT (((SCONV_IN_HEIGHT) - (SCONV_KERNEL_HEIGHT)) / (SCONV_STRIDE) + 1)
#define SCONV_OUT_WIDTH  (((SCONV_IN_WIDTH)  - (SCONV_KERNEL_WIDTH))  / (SCONV_STRIDE) + 1)

// How many column positions of one row the line buffer keeps addressable at
// once. Exactly the kernel width is the cheapest value that still gives
// every column in a window its own bank -- same reasoning as conv2d.hpp's
// CONV_IN_COL_BANKS.
#define SCONV_COL_BANKS (SCONV_KERNEL_WIDTH)

// =====================================================================
// 3. Compile-time sanity checks
// =====================================================================
static_assert(SCONV_STRIDE >= 1,                    "stride must be at least 1");
static_assert(SCONV_IN_HEIGHT >= SCONV_KERNEL_HEIGHT, "kernel is taller than the input");
static_assert(SCONV_IN_WIDTH  >= SCONV_KERNEL_WIDTH,  "kernel is wider than the input");
static_assert(SCONV_OUT_HEIGHT >= 1 && SCONV_OUT_WIDTH >= 1, "output would be empty");

namespace sconv_detail {
constexpr long long kHalfRange = 1LL << (8 * sizeof(data_t) - 1);
constexpr long long kWorstCase =
    (long long)SCONV_IN_CHANNELS * SCONV_KERNEL_HEIGHT * SCONV_KERNEL_WIDTH * kHalfRange * kHalfRange;
}
static_assert(sconv_detail::kWorstCase > 0, "result_t may be too narrow for this configuration");

// =====================================================================
// 4. Pixel types
//
//    One struct = every channel of one pixel, so one memory word carries
//    a whole pixel. This is the layout change that makes streaming work:
//    conv2d.hpp's arrays are channel-major (data_t[channel][row][col]), so
//    reading "all channels at this row,col" means several separate BRAM
//    accesses. Here it's channel-last / channel-interleaved
//    (pixel[row][col].channel[c]) so it's a single read. See part 6.
// =====================================================================
struct sconv_in_pixel_t  { data_t   channel[SCONV_IN_CHANNELS]; };
struct sconv_out_pixel_t { result_t channel[SCONV_OUT_CHANNELS]; };

// =====================================================================
// 5. Plain constexpr mirrors, for testbench / host code.
// =====================================================================
namespace sconv {
constexpr int in_channels = SCONV_IN_CHANNELS, out_channels = SCONV_OUT_CHANNELS;
constexpr int in_height   = SCONV_IN_HEIGHT,   in_width     = SCONV_IN_WIDTH;
constexpr int kernel_height = SCONV_KERNEL_HEIGHT, kernel_width = SCONV_KERNEL_WIDTH;
constexpr int out_height  = SCONV_OUT_HEIGHT,  out_width    = SCONV_OUT_WIDTH;
constexpr int stride = SCONV_STRIDE;
}

// =====================================================================
// 6. What "streaming" buys, and what it does not (yet)
//
//    conv2d.cpp copies the whole input image into on-chip BRAM before it
//    computes anything: IN_CHANNELS x IN_HEIGHT x IN_WIDTH x 2 bytes. At
//    28x28x3 that's under 5 KB, fine. At a real hyperspectral frame
//    (roughly 500x300x104) it's ~30 MB -- far more on-chip memory than any
//    board here has.
//
//    This kernel never holds the whole image. It keeps only
//    SCONV_KERNEL_HEIGHT rows (a handful, not hundreds), so its on-chip
//    footprint depends on image WIDTH and KERNEL size, never on image
//    HEIGHT. That is the entire trick, and it is why SCONV_IN_HEIGHT can
//    grow freely without the design changing shape.
//
//    What this version does NOT yet do: every input and output channel is
//    computed in parallel, every cycle (SCONV_IN_CHANNELS x
//    SCONV_OUT_CHANNELS x KERNEL_HEIGHT x KERNEL_WIDTH multipliers). That
//    is fine at 3 in / 3 out channels. It is not fine at 104 input
//    channels -- conv2d.hpp's CONV_CHANNELS_PER_PASS trick (processing
//    channels in smaller chunks over several cycles instead of all at
//    once) still applies here and still needs adding before this kernel
//    is pointed at a real channel count; it is left out of this first
//    version so the streaming part could be gotten right and tested on
//    its own first.
// =====================================================================

void streaming_conv2d(
    const sconv_in_pixel_t input_image [SCONV_IN_HEIGHT][SCONV_IN_WIDTH],
    const data_t           kernel_weights[SCONV_OUT_CHANNELS][SCONV_IN_CHANNELS]
                                         [SCONV_KERNEL_HEIGHT][SCONV_KERNEL_WIDTH],
    sconv_out_pixel_t      output_image[SCONV_OUT_HEIGHT][SCONV_OUT_WIDTH]);

#endif // STREAMING_CONV2D_HPP
