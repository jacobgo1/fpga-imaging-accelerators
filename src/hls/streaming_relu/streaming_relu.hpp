#ifndef STREAMING_RELU_HPP
#define STREAMING_RELU_HPP

#include <cstdint>
#include <limits>

// =====================================================================
// 1. Data types
//
//    ReLU sits right after conv2d in a real pipeline, so its input is
//    conv2d's wide accumulator type (result_t) -- and its output is the
//    narrow type (data_t) everything downstream (pooling, the next conv's
//    weights) actually uses. So this module does two jobs in one pass:
//    zero out negative values, AND narrow 64 bits back down to 16 before
//    passing the pixel on. See part 4 for why narrowing needs care.
// =====================================================================
using data_t   = std::int16_t;
using result_t = std::int64_t;

// =====================================================================
// 2. Layer geometry
//
//    Prefixed SRELU_, same reasoning as every other streaming_* header:
//    macros are global text substitution, so each kernel gets its own
//    prefix to avoid collisions if two are ever compiled together.
//
//    Default size here matches streaming_conv2d's default output
//    (26x26, 3 channels) -- this module reads as "the next stage after
//    that one," even though nothing is wired together yet.
// =====================================================================
#define SRELU_CHANNELS   3
#define SRELU_HEIGHT    26
#define SRELU_WIDTH     26

// =====================================================================
// 3. Pixel types -- channel-last, same idea as every other streaming_*
//    header: one struct holds every channel of one pixel.
// =====================================================================
struct srelu_in_pixel_t  { result_t channel[SRELU_CHANNELS]; };
struct srelu_out_pixel_t { data_t   channel[SRELU_CHANNELS]; };

// =====================================================================
// 4. Plain constexpr mirrors, for testbench / host code.
// =====================================================================
namespace srelu {
constexpr int channels = SRELU_CHANNELS;
constexpr int height   = SRELU_HEIGHT;
constexpr int width    = SRELU_WIDTH;
}

// =====================================================================
// 5. Why this needs no line buffer, and why narrowing needs a clamp
//
//    ReLU looks at one value and decides "keep it or zero it" -- it never
//    compares a pixel against its neighbours the way conv2d and pooling
//    do. So there is nothing to remember between pixels at all: no line
//    buffer, no fill-up period, not even one row of history. Every output
//    pixel depends on exactly one input pixel. This is the simplest of the
//    four streaming modules for exactly that reason.
//
//    The narrowing is the part that needs care. conv2d.hpp's own
//    worst-case check shows a 64-bit accumulator can legitimately hold a
//    value far larger than a 16-bit type can represent. Blindly truncating
//    (data_t)big_value would silently wrap around to a small or negative
//    number -- a real, silent correctness bug, not a hypothetical one.
//    So instead of truncating, this module clamps ("saturates"): any
//    positive value too big for data_t becomes the largest value data_t
//    CAN hold, rather than wrapping.
// =====================================================================

void streaming_relu(
    const srelu_in_pixel_t  input_image [SRELU_HEIGHT][SRELU_WIDTH],
    srelu_out_pixel_t       output_image[SRELU_HEIGHT][SRELU_WIDTH]);

#endif // STREAMING_RELU_HPP
