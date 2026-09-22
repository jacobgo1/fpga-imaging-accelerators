#ifndef STREAMING_UPSAMPLE_HPP
#define STREAMING_UPSAMPLE_HPP

#include <cstdint>

// =====================================================================
// 1. Data type
//
//    Nearest-neighbor upsampling only ever copies a value -- it never
//    does arithmetic on it -- so, like pooling, it needs only one type,
//    the narrow one everything between conv stages uses.
// =====================================================================
using data_t = std::int16_t;

// =====================================================================
// 2. Layer geometry
//
//    Prefixed SUPSAMPLE_, same reasoning as every streaming_* header.
//
//    Default size here follows on from streaming_maxpool's default output
//    (13x13, 3 channels), upsampled by the same factor (2) that pool used
//    to shrink it -- the shape a decoder stage undoing that pool would see,
//    even though nothing is wired together yet.
// =====================================================================
#define SUPSAMPLE_CHANNELS    3
#define SUPSAMPLE_IN_HEIGHT  13   // input rows -- can be large; memory cost doesn't care
#define SUPSAMPLE_IN_WIDTH   13   // input columns
#define SUPSAMPLE_FACTOR      2   // each input pixel becomes a FACTOR x FACTOR block

#define SUPSAMPLE_OUT_HEIGHT ((SUPSAMPLE_IN_HEIGHT) * (SUPSAMPLE_FACTOR))
#define SUPSAMPLE_OUT_WIDTH  ((SUPSAMPLE_IN_WIDTH)  * (SUPSAMPLE_FACTOR))

// =====================================================================
// 3. Compile-time sanity checks
// =====================================================================
static_assert(SUPSAMPLE_FACTOR >= 1, "upsampling factor must be at least 1");

// =====================================================================
// 4. Pixel type -- channel-last, same idea as every other streaming_*
//    header: one struct holds every channel of one pixel.
// =====================================================================
struct supsample_pixel_t { data_t channel[SUPSAMPLE_CHANNELS]; };

// =====================================================================
// 5. Plain constexpr mirrors, for testbench / host code.
// =====================================================================
namespace supsample {
constexpr int channels  = SUPSAMPLE_CHANNELS;
constexpr int in_height = SUPSAMPLE_IN_HEIGHT, in_width  = SUPSAMPLE_IN_WIDTH;
constexpr int out_height = SUPSAMPLE_OUT_HEIGHT, out_width = SUPSAMPLE_OUT_WIDTH;
constexpr int factor = SUPSAMPLE_FACTOR;
}

// =====================================================================
// 6. What this buffers, and why it's less than conv/pool need
//
//    conv2d and maxpool need several ROWS of history at once, because one
//    output pixel is built from a 2D neighbourhood spanning multiple rows.
//    Upsampling never does that -- every output pixel is a plain copy of
//    exactly one input pixel. The only reason it needs to remember
//    anything at all is that each input ROW has to be replayed FACTOR
//    times in a row (to produce FACTOR output rows), and the input only
//    streams past once. So this module keeps exactly one row -- not
//    several, and not a sliding window of one -- and reads it back
//    FACTOR-1 extra times before moving on to the next input row.
// =====================================================================

void streaming_upsample(
    const supsample_pixel_t input_image [SUPSAMPLE_IN_HEIGHT][SUPSAMPLE_IN_WIDTH],
    supsample_pixel_t       output_image[SUPSAMPLE_OUT_HEIGHT][SUPSAMPLE_OUT_WIDTH]);

#endif // STREAMING_UPSAMPLE_HPP
