#ifndef JUSTOUNETSIMPLE_OPT_HLS_HPP
#define JUSTOUNETSIMPLE_OPT_HLS_HPP

#include <stdint.h>
#include "../../optimized/hls_vector_shim.hpp"

// The quantized 2D-JustoUNet (the same model as src/hls/justounetsimple/, the
// golden baseline), built for speed: a DATAFLOW chain of processes that all run
// at the same time, connected by streams, processing n patches per call.
//
//   load ─► conv1 ─► pool ─► conv2 ─► pool+up ─► conv3 ─► up ─► conv4 ─► store
//   DDR     block A          block B              block C       block D    DDR
//
// While conv1 works on patch k, conv2 works on patch k-1, and so on. The
// slowest process sets the pace: conv1 (110 -> 6 channels), 16 bands per
// cycle, 7 x 32 x 32 + 33 cycles per patch. Everything after it takes all of a
// pixel's channels at once, in raster order, and spreads its output channels
// over several cycles (it has time to spare), so nothing between the blocks
// waits for a whole image.
//
// Integer fixed point throughout (src/optimized/conv3x3_stream.hpp): int16
// activations, the int8 weights as they are, int32 sums. The host converts at
// both ends:
//   din:  round(x * 2^IN_FRAC), saturated to int16   (x = the z-scored band)
//   dout: int32 / 2^OUT_FRAC                         (the scores)
// On the aegean capture this is within 0.004 of the float model's scores and
// gives the same class for 99.998% of pixels (tools/justounetsimple_model.py
// has the bit-exact numpy version: forward_patch_fixed).
//
// din: the 110 bands are padded with zero bands to IN_PAD = 112, 7 chunks of
// 16. A chunk is 16 x 16 bits = 256 bits per pixel: more than one 128-bit HP
// port carries per cycle, so it comes in over two -- bands 0-7 of each chunk
// through din0, bands 8-15 through din1, each its own m_axi bundle and HP port.
// Per port, per patch: [IN_PAD/P1][H][W][PORT_BANDS] int16, plane by plane,
// the order conv1 consumes them, so no patch is stored on chip. The host
// builds both from an n x H x W x IN_PAD batch with
//   batch.reshape(n, H, W, IN_PAD/P1, 2, PORT_BANDS).transpose(4, 0, 3, 1, 2, 5)
// (index 0 of the result is din0, index 1 din1).
// dout layout: [n][H][W][OUT_CH] int32 (HWC, as the golden kernel's floats).
// din must already be preprocessed (band selection and z-score).

#define JOPT_H           32
#define JOPT_W           32
#define JOPT_IN_CH       110   // bands the model uses
#define JOPT_BASE_CH     6
#define JOPT_OUT_CH      3
#define JOPT_P1          16    // bands per cycle into conv1
#define JOPT_PORT_BANDS  8     // bands per 128-bit word, on each of the two input ports
// IN_CH rounded up to a multiple of P1: 112 bands in DDR, the last 2 zero.
#define JOPT_IN_PAD      ((JOPT_IN_CH + JOPT_P1 - 1) / JOPT_P1 * JOPT_P1)

#define JOPT_IN_FRAC     11    // din  = x * 2^11 (int16: -16 .. +16)
#define JOPT_ACT_FRAC    11    // activations between the blocks, the same
#define JOPT_OUT_FRAC    16    // dout = score * 2^16

// One port's word: 8 bands of a pixel, 128 bits.
typedef hls::vector<int16_t, JOPT_PORT_BANDS> jopt_in_t;

void justounetsimple_opt(const jopt_in_t *din0, const jopt_in_t *din1, int32_t *dout, int n);

#endif // JUSTOUNETSIMPLE_OPT_HLS_HPP
