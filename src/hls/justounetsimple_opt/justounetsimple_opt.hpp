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
// While block A works on patch k, block B works on patch k-1, and so on. The
// slowest stage sets the pace: block A, conv1 (110 -> 6 channels), at about
// 55 chunks x 33 x 33 cycles per patch. The other blocks have II > 1 so they
// share their multipliers and still keep up.
//
// din layout in DDR, per patch: [IN_CH/P1][H][W][P1] -- P1 bands at a time,
// plane by plane, exactly the order block A consumes them, so no patch is
// stored on chip. The host builds it from an H x W x IN_CH patch with
// patch.reshape(H, W, IN_CH/P1, P1).transpose(2, 0, 1, 3).
// dout layout: [n][H][W][OUT_CH] (HWC, the same as the golden kernel).
// din must already be preprocessed (band selection and z-score).

#define JOPT_H        32
#define JOPT_W        32
#define JOPT_IN_CH    110
#define JOPT_BASE_CH  6
#define JOPT_OUT_CH   3
#define JOPT_P1       2     // input channels per cycle in block A

// One 2-band vector of the input: a single 64-bit word on the AXI bus.
typedef hls::vector<float, JOPT_P1> jopt_in_t;

void justounetsimple_opt(const jopt_in_t *din, float *dout, int n);

#endif // JUSTOUNETSIMPLE_OPT_HLS_HPP
