#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <cstdint>

// =====================================================================
// 1. Data types
// =====================================================================
using data_t   = std::int16_t;   // input pixels and weights
using result_t = std::int64_t;   // accumulator and output

// =====================================================================
// 2. Layer geometry  --  the only block you normally edit
//
//    These are #defines rather than constexpr on purpose: they are used
//    inside #pragma arguments, and pragmas are parsed from the text the
//    *preprocessor* produces. Macros always work there; constexpr works
//    only sometimes, depending on the Vitis version and the position.
//
//    Prefixed CONV_ because macros are global text substitution with no
//    namespace -- an unprefixed WIDTH or STRIDE could silently collide
//    with another kernel's macro if the two are ever compiled together.
// =====================================================================
#define CONV_IN_CHANNELS       3   // input channels (e.g. spectral bands)
#define CONV_OUT_CHANNELS      3   // output channels (learned feature maps)
#define CONV_IN_HEIGHT        28   // input rows
#define CONV_IN_WIDTH         28   // input columns
#define CONV_KERNEL_HEIGHT     3   // convolution window rows
#define CONV_KERNEL_WIDTH      3   // convolution window columns
#define CONV_STRIDE             1   // window step per output pixel; no padding ("valid")
#define CONV_CHANNELS_PER_PASS  3   // input channels accumulated together per pass, see part 6 below

// Derived output geometry. Every sub-expression is parenthesised because
// macros are textual substitution -- without the parens, passing
// something like (2+1) as CONV_STRIDE would silently compute nonsense.
#define CONV_OUT_HEIGHT  (((CONV_IN_HEIGHT) - (CONV_KERNEL_HEIGHT)) / (CONV_STRIDE) + 1)
#define CONV_OUT_WIDTH   (((CONV_IN_WIDTH)  - (CONV_KERNEL_WIDTH))  / (CONV_STRIDE) + 1)

// =====================================================================
// 3. Memory banking factors
//
//    These are what make the II=1 schedule possible. They must be at
//    least the kernel size; exactly the kernel size is the cheapest
//    value that still removes every port conflict, so tie them to it.
// =====================================================================
#define CONV_IN_ROW_BANKS  (CONV_KERNEL_HEIGHT)
#define CONV_IN_COL_BANKS  (CONV_KERNEL_WIDTH)

// =====================================================================
// 4. Compile-time sanity checks
//    These fire during C simulation, long before you waste 20 minutes
//    on a synthesis run that was doomed from the start.
// =====================================================================
static_assert(CONV_STRIDE >= 1,             "stride must be at least 1");
static_assert(CONV_IN_HEIGHT >= CONV_KERNEL_HEIGHT, "kernel is taller than the input");
static_assert(CONV_IN_WIDTH  >= CONV_KERNEL_WIDTH,  "kernel is wider than the input");
static_assert(CONV_OUT_HEIGHT >= 1 && CONV_OUT_WIDTH >= 1, "output would be empty");
static_assert(CONV_IN_ROW_BANKS >= CONV_KERNEL_HEIGHT, "row banks < kernel height: II=1 is impossible");
static_assert(CONV_IN_COL_BANKS >= CONV_KERNEL_WIDTH,  "col banks < kernel width: II=1 is impossible");
static_assert(CONV_IN_CHANNELS % CONV_CHANNELS_PER_PASS == 0,
             "CONV_CHANNELS_PER_PASS must divide CONV_IN_CHANNELS");

// Accumulator width check: worst-case |sum| for the chosen types/sizes.
namespace conv_detail {
constexpr long long kHalfRange = 1LL << (8 * sizeof(data_t) - 1);
constexpr long long kWorstCase =
    (long long)CONV_IN_CHANNELS * CONV_KERNEL_HEIGHT * CONV_KERNEL_WIDTH * kHalfRange * kHalfRange;
}
static_assert(conv_detail::kWorstCase > 0, "result_t may be too narrow for this configuration");

// =====================================================================
// 5. Plain constexpr mirrors, for testbench / host code that does not
//    want to shout in macros. Not usable inside pragmas.
// =====================================================================
namespace conv {
constexpr int in_channels = CONV_IN_CHANNELS, out_channels = CONV_OUT_CHANNELS;
constexpr int in_height   = CONV_IN_HEIGHT,   in_width     = CONV_IN_WIDTH;
constexpr int kernel_height = CONV_KERNEL_HEIGHT, kernel_width = CONV_KERNEL_WIDTH;
constexpr int out_height  = CONV_OUT_HEIGHT,  out_width    = CONV_OUT_WIDTH;
constexpr int stride = CONV_STRIDE;
}

// =====================================================================
// 6. Why there's a "pass" at all
//
//    CONV_CHANNELS_PER_PASS input channels are summed in parallel, in one
//    cycle, by fully unrolled hardware (see conv2d.cpp). If it equals
//    CONV_IN_CHANNELS, every channel is summed at once and there is only
//    one pass. Set it lower to trade latency for DSPs/registers: fewer
//    channels in parallel means less hardware, but the chip has to make
//    CONV_IN_CHANNELS / CONV_CHANNELS_PER_PASS passes over every output
//    pixel, adding the partial sums from each pass together.
// =====================================================================

void conv2d(const data_t input_image [CONV_IN_CHANNELS][CONV_IN_HEIGHT][CONV_IN_WIDTH],
            const data_t kernel_weights[CONV_OUT_CHANNELS][CONV_IN_CHANNELS]
                                       [CONV_KERNEL_HEIGHT][CONV_KERNEL_WIDTH],
            result_t     output_image[CONV_OUT_CHANNELS][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]);

#endif // CONV2D_HPP
