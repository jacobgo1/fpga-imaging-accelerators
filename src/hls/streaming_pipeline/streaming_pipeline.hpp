#ifndef STREAMING_PIPELINE_HPP
#define STREAMING_PIPELINE_HPP

#include "streaming_conv2d.hpp"
#include "streaming_relu.hpp"
#include "streaming_maxpool.hpp"

// =====================================================================
// 1. What this kernel is, and what it is not
//
//    streaming_conv2d, streaming_relu and streaming_maxpool each already
//    avoid holding a whole image on-chip *inside themselves* (see each
//    header's own part on the topic). Now that all three take
//    hls::stream ports instead of arrays (streaming_conv2d.hpp part 6),
//    this kernel is just those three functions -- called directly, not
//    reimplemented -- wired together by hls::stream FIFOs (a handful of
//    pixels deep, not a frame) under "#pragma HLS DATAFLOW" so the three
//    run concurrently: maxpool is consuming row 3 of relu's output while
//    conv2d is still producing row 9. See streaming_pipeline.cpp.
//
//    This does not define its own geometry. SCONV_*/SRELU_*/SPOOL_* (from
//    the three headers above) already say what sizes this pipeline runs
//    at -- repeating them here as a fourth, SPIPE_-prefixed set would be
//    exactly the kind of duplication that makes it easy for the pipeline
//    and the modules it calls to quietly drift apart. Instead, part 2
//    below checks at compile time that streaming_conv2d's output shape
//    really is streaming_relu's input shape, and so on down the chain --
//    so a size mismatch between modules is a build error here, not a
//    silently wrong simulation.
//
//    What this does NOT do: batch several such pipelines, or handle more
//    than one conv+relu+pool stage. It is a sketch of the wiring pattern.
//    Extending it to a real multi-layer network means repeating this
//    stream/DATAFLOW pattern once per layer.
// =====================================================================

// =====================================================================
// 2. Cross-module geometry checks
//
//    Each module still has its own independent macros (see every
//    streaming_*.hpp's own part on macro collisions) sized to chain after
//    one another by convention, not by construction. These static_asserts
//    make that convention a compile-time guarantee instead of a comment:
//    change one module's geometry without updating the others it is
//    chained with here, and this header -- not a mismatched loop deep
//    inside streaming_pipeline.cpp -- is what fails to compile.
// =====================================================================
static_assert(SCONV_OUT_HEIGHT   == SRELU_HEIGHT,     "conv2d output height must match relu's input height");
static_assert(SCONV_OUT_WIDTH    == SRELU_WIDTH,      "conv2d output width must match relu's input width");
static_assert(SCONV_OUT_CHANNELS == SRELU_CHANNELS,   "conv2d output channels must match relu's channel count");
static_assert(SRELU_HEIGHT       == SPOOL_IN_HEIGHT,  "relu output height must match maxpool's input height");
static_assert(SRELU_WIDTH        == SPOOL_IN_WIDTH,   "relu output width must match maxpool's input width");
static_assert(SRELU_CHANNELS     == SPOOL_CHANNELS,   "relu output channels must match maxpool's channel count");

// The pipeline's own input/output boundary sizes, purely for readability
// at the call site (main.py/testbench) -- not a separate source of truth,
// just names for streaming_conv2d's input geometry and streaming_maxpool's
// output geometry.
#define SPIPE_IN_HEIGHT    SCONV_IN_HEIGHT
#define SPIPE_IN_WIDTH     SCONV_IN_WIDTH
#define SPIPE_IN_CHANNELS  SCONV_IN_CHANNELS
#define SPIPE_OUT_CHANNELS SCONV_OUT_CHANNELS
#define SPIPE_KERNEL_HEIGHT SCONV_KERNEL_HEIGHT
#define SPIPE_KERNEL_WIDTH  SCONV_KERNEL_WIDTH
#define SPIPE_OUT_HEIGHT   SPOOL_OUT_HEIGHT
#define SPIPE_OUT_WIDTH    SPOOL_OUT_WIDTH

// =====================================================================
// 3. Plain constexpr mirrors, for testbench / host code.
// =====================================================================
namespace spipe {
constexpr int in_channels = SPIPE_IN_CHANNELS, out_channels = SPIPE_OUT_CHANNELS;
constexpr int in_height   = SPIPE_IN_HEIGHT,   in_width     = SPIPE_IN_WIDTH;
constexpr int kernel_height = SPIPE_KERNEL_HEIGHT, kernel_width = SPIPE_KERNEL_WIDTH;
constexpr int stride = SCONV_STRIDE;
constexpr int conv_out_height = SCONV_OUT_HEIGHT, conv_out_width = SCONV_OUT_WIDTH;
constexpr int pool_kernel_height = SPOOL_KERNEL_HEIGHT, pool_kernel_width = SPOOL_KERNEL_WIDTH;
constexpr int pool_stride = SPOOL_STRIDE;
constexpr int out_height = SPIPE_OUT_HEIGHT, out_width = SPIPE_OUT_WIDTH;
}

// input_image uses streaming_conv2d's own input pixel type; output_image
// uses streaming_maxpool's own (pooling never changes the pixel type) --
// reused directly, not redeclared, so there is exactly one definition of
// "what a conv2d input pixel looks like" for this whole chain.
void streaming_pipeline(
    const sconv_in_pixel_t input_image [SPIPE_IN_HEIGHT][SPIPE_IN_WIDTH],
    const data_t           kernel_weights[SPIPE_OUT_CHANNELS][SPIPE_IN_CHANNELS]
                                         [SPIPE_KERNEL_HEIGHT][SPIPE_KERNEL_WIDTH],
    spool_pixel_t          output_image[SPIPE_OUT_HEIGHT][SPIPE_OUT_WIDTH]);

#endif // STREAMING_PIPELINE_HPP
