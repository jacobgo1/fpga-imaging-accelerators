#include "streaming_pipeline.hpp"

// =====================================================================
// This is nothing but wiring: streaming_conv2d, streaming_relu and
// streaming_maxpool are called directly below -- the actual, already-
// tested top functions from their own kernels, unmodified in behavior --
// and connected through hls::stream FIFOs under #pragma HLS DATAFLOW so
// the three run concurrently. See streaming_pipeline.hpp parts 1-2 for
// why that beats three separate accelerator calls glued together with
// full-frame arrays, and for how the geometry is kept from drifting
// between them.
//
// This only works because those three kernels now take hls::stream ports
// instead of arrays (streaming_conv2d.hpp part 6) -- that is the change
// this pipeline required of the existing modules, and the only one: their
// algorithms did not change. Because input_image/output_image below use
// streaming_conv2d's and streaming_maxpool's own pixel types directly
// (see streaming_pipeline.hpp part 3), the feed/drain loops are pure data
// movement, no conversion.
//
// Between stages it's a different story: each module keeps its own
// independently-named pixel struct (sconv_out_pixel_t, srelu_in_pixel_t, ...)
// on purpose -- see every streaming_*.hpp's part 4/5 -- so any one of them
// can be swapped for a differently-shaped replacement without the others
// needing to change. At this pipeline's default sizes those structs are
// layout-identical (same channel count, same element type) but still
// distinct C++ types, so relu_stage_input()/maxpool_stage_input() below
// exist purely to repackage one into the other, one channel at a time --
// no arithmetic, nothing recomputed.
// =====================================================================

// sconv_out_pixel_t (conv's output) -> srelu_in_pixel_t (relu's input).
static void relu_stage_input(hls::stream<sconv_out_pixel_t>& conv_out,
                              hls::stream<srelu_in_pixel_t>&  relu_in)
{
    relabel_conv_to_relu:
    for (int row = 0; row < SCONV_OUT_HEIGHT; row++)
        for (int col = 0; col < SCONV_OUT_WIDTH; col++) {
            #pragma HLS PIPELINE II=1
            const sconv_out_pixel_t pixel = conv_out.read();
            srelu_in_pixel_t relabeled;
            for (int c = 0; c < SCONV_OUT_CHANNELS; c++) {
                #pragma HLS UNROLL
                relabeled.channel[c] = pixel.channel[c];
            }
            relu_in.write(relabeled);
        }
}

// srelu_out_pixel_t (relu's output) -> spool_pixel_t (maxpool's input).
static void maxpool_stage_input(hls::stream<srelu_out_pixel_t>& relu_out,
                                 hls::stream<spool_pixel_t>&     pool_in)
{
    relabel_relu_to_maxpool:
    for (int row = 0; row < SRELU_HEIGHT; row++)
        for (int col = 0; col < SRELU_WIDTH; col++) {
            #pragma HLS PIPELINE II=1
            const srelu_out_pixel_t pixel = relu_out.read();
            spool_pixel_t relabeled;
            for (int c = 0; c < SRELU_CHANNELS; c++) {
                #pragma HLS UNROLL
                relabeled.channel[c] = pixel.channel[c];
            }
            pool_in.write(relabeled);
        }
}

void streaming_pipeline(
    const sconv_in_pixel_t input_image [SPIPE_IN_HEIGHT][SPIPE_IN_WIDTH],
    const data_t           kernel_weights[SPIPE_OUT_CHANNELS][SPIPE_IN_CHANNELS]
                                         [SPIPE_KERNEL_HEIGHT][SPIPE_KERNEL_WIDTH],
    spool_pixel_t          output_image[SPIPE_OUT_HEIGHT][SPIPE_OUT_WIDTH])
{
    #pragma HLS INTERFACE mode=bram port=input_image
    #pragma HLS INTERFACE mode=bram port=kernel_weights
    #pragma HLS INTERFACE mode=bram port=output_image
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    #pragma HLS DATAFLOW

    hls::stream<sconv_in_pixel_t>  conv_in ("conv_in");
    hls::stream<sconv_out_pixel_t> conv_out("conv_out");
    hls::stream<srelu_in_pixel_t>  relu_in ("relu_in");
    hls::stream<srelu_out_pixel_t> relu_out("relu_out");
    hls::stream<spool_pixel_t>     pool_in ("pool_in");
    hls::stream<spool_pixel_t>     pool_out("pool_out");
    // A few pixels deep, not a frame -- this is the entire point. Depth is
    // a throughput/overlap tuning knob (how far stages can drift apart
    // before one stalls the other), never a correctness requirement for a
    // single linear producer/consumer chain like this one.
    #pragma HLS STREAM variable=conv_in  depth=4
    #pragma HLS STREAM variable=conv_out depth=4
    #pragma HLS STREAM variable=relu_in  depth=4
    #pragma HLS STREAM variable=relu_out depth=4
    #pragma HLS STREAM variable=pool_in  depth=4
    #pragma HLS STREAM variable=pool_out depth=4

    // ---- feed: the pipeline's true input boundary is this array; every
    // stage after this one only ever touches a stream. ----
    feed_input:
    for (int row = 0; row < SPIPE_IN_HEIGHT; row++)
        for (int col = 0; col < SPIPE_IN_WIDTH; col++) {
            #pragma HLS PIPELINE II=1
            conv_in.write(input_image[row][col]);
        }

    streaming_conv2d(conv_in, kernel_weights, conv_out);
    relu_stage_input(conv_out, relu_in);
    streaming_relu(relu_in, relu_out);
    maxpool_stage_input(relu_out, pool_in);
    streaming_maxpool(pool_in, pool_out);

    // ---- drain: the pipeline's true output boundary. ----
    drain_output:
    for (int row = 0; row < SPIPE_OUT_HEIGHT; row++)
        for (int col = 0; col < SPIPE_OUT_WIDTH; col++) {
            #pragma HLS PIPELINE II=1
            output_image[row][col] = pool_out.read();
        }
}
