#ifndef STREAMING_MAXPOOL_HPP
#define STREAMING_MAXPOOL_HPP

#include <cstdint>
#include <limits>

// =====================================================================
// 1. Data type
//
//    Max picks one of its inputs back out -- it never adds anything up --
//    so the output is never wider than the input. Unlike streaming_conv2d
//    (which widens to a 64-bit accumulator because summing many products
//    can overflow), pooling needs only one type.
// =====================================================================
using data_t = std::int16_t;

// =====================================================================
// 2. Layer geometry
//
//    Prefixed SPOOL_ so this kernel's macros can't collide with conv2d's
//    CONV_ or streaming_conv2d's SCONV_ ones if they're ever compiled
//    together -- same reasoning as those headers.
//
//    Test config here follows on from streaming_conv2d's own test output
//    (26x26, 3 channels) with the classic non-overlapping 2x2/stride-2
//    pool, so it reads as "the next stage after that one," even though the
//    two aren't wired together yet.
// =====================================================================
#define SPOOL_CHANNELS         3   // channels; pooling never mixes channels together
#define SPOOL_IN_HEIGHT       26   // input rows -- can be large; memory cost doesn't care
#define SPOOL_IN_WIDTH        26   // input columns
#define SPOOL_KERNEL_HEIGHT    2   // pooling window rows
#define SPOOL_KERNEL_WIDTH     2   // pooling window columns
#define SPOOL_STRIDE            2   // usually == kernel size (non-overlapping pooling)

#define SPOOL_OUT_HEIGHT (((SPOOL_IN_HEIGHT) - (SPOOL_KERNEL_HEIGHT)) / (SPOOL_STRIDE) + 1)
#define SPOOL_OUT_WIDTH  (((SPOOL_IN_WIDTH)  - (SPOOL_KERNEL_WIDTH))  / (SPOOL_STRIDE) + 1)

// Same reasoning as streaming_conv2d.hpp's SCONV_COL_BANKS: enough column
// banks in the line buffer to give every column a window might touch its
// own bank, so all of them can be read in the same cycle.
#define SPOOL_COL_BANKS (SPOOL_KERNEL_WIDTH)

// =====================================================================
// 3. Compile-time sanity checks
// =====================================================================
static_assert(SPOOL_STRIDE >= 1,                    "stride must be at least 1");
static_assert(SPOOL_IN_HEIGHT >= SPOOL_KERNEL_HEIGHT, "kernel is taller than the input");
static_assert(SPOOL_IN_WIDTH  >= SPOOL_KERNEL_WIDTH,  "kernel is wider than the input");
static_assert(SPOOL_OUT_HEIGHT >= 1 && SPOOL_OUT_WIDTH >= 1, "output would be empty");

// =====================================================================
// 4. Pixel type -- channel-last, same idea as streaming_conv2d.hpp's
//    sconv_in_pixel_t: one struct holds every channel of one pixel, so one
//    memory word carries a whole pixel and a single read gets all of it.
// =====================================================================
struct spool_pixel_t { data_t channel[SPOOL_CHANNELS]; };

// =====================================================================
// 5. Plain constexpr mirrors, for testbench / host code.
// =====================================================================
namespace spool {
constexpr int channels = SPOOL_CHANNELS;
constexpr int in_height = SPOOL_IN_HEIGHT, in_width  = SPOOL_IN_WIDTH;
constexpr int kernel_height = SPOOL_KERNEL_HEIGHT, kernel_width = SPOOL_KERNEL_WIDTH;
constexpr int out_height = SPOOL_OUT_HEIGHT, out_width = SPOOL_OUT_WIDTH;
constexpr int stride = SPOOL_STRIDE;
}

void streaming_maxpool(
    const spool_pixel_t input_image [SPOOL_IN_HEIGHT][SPOOL_IN_WIDTH],
    spool_pixel_t       output_image[SPOOL_OUT_HEIGHT][SPOOL_OUT_WIDTH]);

#endif // STREAMING_MAXPOOL_HPP
