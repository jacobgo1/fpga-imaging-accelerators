#ifndef JUSTOUNETSIMPLE_OPT_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_HLS_HPP

#include "../../optimized/stream_ops.hpp"
#include "../../optimized/hls_vector_shim.hpp"

// The quantized 2D-JustoUNet (the same model as src/hls/justounetsimple/, the
// golden baseline), built for speed: a DATAFLOW chain of six stages that all
// run at the same time, connected by streams, processing n patches per call.
//
//   load ─► block A ─► block B ─► block C ─► block D ─► store
//   DDR     conv1       conv2       conv3       conv4      DDR
//           ReLU,pool   ReLU,pool,  ReLU,up
//                       up
//
// While block A works on patch k, block B works on patch k-1, and so on.
// Each block reads its P input channels at a time (P1 for block A, P2 for
// B, P3 for C, P4 for D) -- that is the knob that trades DSPs for cycles:
// block N takes roughly (its IN_CH / P) x (its H+1) x (its W+1) x II cycles
// per patch. Block A has by far the most input channels (112, padded from
// the model's 110 so it divides P1 evenly), so it was the pipeline's
// bottleneck at P1=2; P1=16 cuts its cycles ~8x. The other three blocks
// still keep up at their lower P2..P4 with II > 1, sharing multipliers.
//
// din layout in DDR, per patch: [IN_CH_PAD/P1][H][W][P1] -- P1 bands at a
// time, plane by plane, exactly the order block A consumes them, so no
// patch is stored on chip. IN_CH_PAD is IN_CH (110 real bands) zero-padded
// up to 112 so P1 divides it evenly; the host must supply those 2 extra
// channels as zero. Build it from an H x W x IN_CH_PAD patch (last 2
// channels zero) with patch.reshape(H, W, IN_CH_PAD/P1, P1).transpose(2, 0, 1, 3).
// dout layout: [n][H][W][OUT_CH] (HWC, the same as the golden kernel).
// din must already be preprocessed (band selection and z-score).

#define JOPT_H            32
#define JOPT_W            32
#define JOPT_IN_CH        110
#define JOPT_IN_CH_PAD    112
#define JOPT_BASE_CH      6
#define JOPT_OUT_CH       3
#define JOPT_P1           16     // input channels per cycle in block A (conv1, 112 ch)
#define JOPT_P2           2      // input channels per cycle in block B (conv2, 6 ch)
#define JOPT_P3           4      // input channels per cycle in block C (conv3, 12 ch)
#define JOPT_P4           2      // input channels per cycle in block D (conv4, 6 ch)

// One P1-band vector of the input: a single wide word on the AXI bus
// (JOPT_P1 x 32 bits).
typedef hls::vector<float, JOPT_P1> jopt_in_t;

void justounetsimple_opt(const jopt_in_t *din, float *dout, int n);

#endif // JUSTOUNETSIMPLE_OPT_HLS_HPP
